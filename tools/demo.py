import importlib.util
import sys
import argparse
import os
import numpy as np
import cv2
import time

# from adaint.model import PARAM_PREDICTOR
# import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmedit.models import build_model

from ailut_demo import ailut_transform as lookup

def nothing(x):
    pass


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class TPAMIBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        pretrained (bool, optional): [ignored].
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to insert an extra pooling layer
            at the very end of the module to reduce the number of parameters of
            the subsequent module. Default: False.
    """

    def __init__(self, pretrained=False, input_resolution=256, extra_pooling=False, norm=True):
        body = [
            BasicBlock(3, 16, stride=2, norm=norm),
            BasicBlock(16, 32, stride=2, norm=norm),
            BasicBlock(32, 64, stride=2, norm=norm),
            BasicBlock(64, 128, stride=2, norm=norm),
            BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
        ]
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class WeightsGenerator(nn.Module):
    r"""The LUT generator module (mapping h).

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks in the mapping h (or the number of basis LUTs).
    """

    def __init__(self, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)

        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.zeros_(self.weights_generator.bias)
        nn.init.zeros_(self.weights_generator.weight)

    def forward(self, x):
        weights = self.weights_generator(x)
        return weights


class Res18Backbone(nn.Module):
    r"""The ResNet-18 backbone.

    Args:
        pretrained (bool, optional): Whether to use the torchvison pretrained weights.
            Default: True.
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 224.
        extra_pooling (bool, optional): [ignore].
    """

    def __init__(self, pretrained=True, input_resolution=224, extra_pooling=False):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)


class IA_NILUT_demo(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3):
        super().__init__()

        self.net = []
        self.net.append(nn.Linear(in_features+3, hidden_features))
        self.net.append(nn.ReLU())

        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)
        self.zero_base = None

    def forward(self, intensity):
        input = torch.cat([intensity, torch.sort(intensity[:,:,:3])[0]], axis=2)

        if self.zero_base is None:
            zero_input = input[:1].clone().detach()
            zero_input[:, :, 3:-3] = 0
            self.zero_base = self.net(zero_input)

        output = self.net(input)
        output = output - self.zero_base + intensity[:,:,:3]
        return output


class DemoEnhancer(nn.Module):
    def __init__(self, backbone='tpami'):
        super().__init__()

        n_colors = 3
        n_vertices = 33
        n_ranks = 5

        self.backbone = dict(
            tpami=TPAMIBackbone,
            res18=Res18Backbone)[backbone.lower()](False, extra_pooling=True)
        self.weights_generator = WeightsGenerator(self.backbone.out_channels, n_ranks)

        self.PG_IA_NILUT = IA_NILUT_demo(in_features=3+n_ranks)
        self.PG_IA_NILUT = self.PG_IA_NILUT.to(torch.float32)

        self.uniform_vertices = torch.arange(n_vertices, dtype=torch.float32).div(n_vertices - 1) \
                                .repeat(n_colors, 1).unsqueeze(0)
        self.identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(n_vertices, dtype=torch.float32) for _ in range(n_colors)]),
                dim=0).div(n_vertices - 1).flip(0),
            *[torch.zeros(
                n_colors, *((n_vertices,) * n_colors), dtype=torch.float32) for _ in range(n_ranks - 1)]
            ], dim=0).view(n_ranks, -1)
        self.identity_lut = self.identity_lut[0].reshape((1, 3, -1))

    def predict_params(self, imgs):
        codes = self.backbone(imgs.float())
        weights = self.weights_generator(codes)
        return weights

    def forward(self, img, param):

        tiled_params = param[:,:,None].repeat([1,1,self.identity_lut.shape[2]])
        ins = torch.cat([self.identity_lut, tiled_params], axis=1)

        ins = torch.permute(ins, (0, 2, 1))
        output_luts = self.PG_IA_NILUT(ins.to(torch.float32))
        output_luts = torch.permute(output_luts, (0, 2, 1))
        output_luts = output_luts.reshape((-1, 3, 33, 33, 33))

        outs = lookup(img.repeat([output_luts.shape[0],1,1,1]).to(torch.float32), output_luts, self.uniform_vertices.repeat([output_luts.shape[0],1,1]))

        return outs


def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cv2.namedWindow('input / output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('parameter', cv2.WINDOW_NORMAL)

    demo_enhancer = DemoEnhancer()
    demo_enhancer.load_state_dict(torch.load(args.checkpoint))

    input_img = cv2.imread(args.img_path)
    input_tensor = torch.permute(torch.tensor(input_img/255.), (2, 0, 1))[None]

    with torch.no_grad():
        parameters = np.array(demo_enhancer.predict_params(input_tensor))[0]

    cv2.namedWindow('input / output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('parameter', cv2.WINDOW_NORMAL)
    parameter_dammy = np.ones((1,400,3)) * (233 / 255)
    previous_parameters = np.copy(parameters)

    filter_names = ["Exposure", "Contrast", "Saturation", "Color temp.", "Tint corr."]

    for i in range(len(filter_names)):
        cv2.createTrackbar(filter_names[i],'parameter',int((parameters[i]+1)*100),200,nothing)

    print("[[Press esc to quit.]]")
    while(1):
        parameters = []
        for i in range(len(filter_names)):
            parameters.append(cv2.getTrackbarPos(filter_names[i], 'parameter')/100.0-1)

        if not np.allclose(parameters, previous_parameters):
            with torch.no_grad():
                output_tensor = demo_enhancer(torch.flip(input_tensor, dims=[1]), torch.tensor([parameters]))
            output_img = np.array(torch.permute(torch.flip(output_tensor[0], dims=[0]), (1, 2, 0)))
            cv2.imshow('input / output', np.hstack([input_img/255., output_img]))
            cv2.imshow('parameter', parameter_dammy)

            previous_parameters = np.copy(parameters)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
