import numbers
import os.path as osp
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import mmcv
from mmcv.runner import auto_fp16

from mmedit.models.base import BaseModel
from mmedit.models.registry import MODELS
from mmedit.models.builder import build_backbone, build_loss
from mmedit.core import psnr, ssim, tensor2img
from mmedit.utils import get_root_logger

from ailut import ailut_transform as lookup


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


class LUTGenerator(nn.Module):
    r"""The LUT generator module (mapping h).

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks in the mapping h (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity


class LUTGenerator(nn.Module):
    r"""The LUT generator module (mapping h).

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks in the mapping h (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(
            self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity




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


class AdaInt(nn.Module):
    r"""The Adaptive Interval Learning (AdaInt) module (mapping g).

    It consists of a single fully-connected layer and some post-process operations.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        adaint_share (bool, optional): Whether to enable Share-AdaInt. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, adaint_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not adaint_share else 1
        self.intervals_generator = nn.Linear(
            n_feats, (n_vertices - 1) * repeat_factor)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.adaint_share = adaint_share

    def init_weights(self):
        r"""Init weights for models.

        We use all-zero and all-one initializations for its weights and bias, respectively.
        """
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, x):
        r"""Forward function for AdaInt module.

        Args:
            x (tensor): Input image representation, shape (b, f).
        Returns:
            Tensor: Sampling coordinates along each lattice dimension, shape (b, c, d).
        """
        x = x.view(x.shape[0], -1)
        intervals = self.intervals_generator(x).view(
            x.shape[0], -1, self.n_vertices - 1)
        if self.adaint_share:
            intervals = intervals.repeat_interleave(self.n_colors, dim=1)
        intervals = intervals.softmax(-1)
        vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        return vertices


@MODELS.register_module()
class AiLUT(BaseModel):
    r"""Adaptive-Interval 3D Lookup Table for real-time image enhancement.

    Args:
        n_ranks (int, optional): Number of ranks in the mapping h
            (or the number of basis LUTs). Default: 3.
        n_vertices (int, optional): Number of sampling points along
            each lattice dimension. Default: 33.
        en_adaint (bool, optional): Whether to enable AdaInt. Default: True.
        en_adaint_share (bool, optional): Whether to enable Share-AdaInt.
            Only used when `en_adaint` is True. Default: False.
        backbone (str, optional): Backbone architecture to use. Can be either 'tpami'
            or 'res18'. Default: 'tpami'.
        pretrained (bool, optional): Whether to use ImageNet-pretrained weights.
            Only used when `backbone` is 'res18'. Default: None.
        n_colors (int, optional): Number of input color channels. Default: 3.
        sparse_factor (float, optional): Loss weight for the sparse regularization term.
            Default: 0.0001.
        smooth_factor (float, optional): Loss weight for the smoothness regularization term.
            Default: 0.
        monotonicity_factor (float, optional): Loss weight for the monotonicaity
            regularization term. Default: 10.0.
        recons_loss (dict, optional): Config for pixel-wise reconstruction loss.
        train_cfg (dict, optional): Config for training. Default: None.
        test_cfg (dict, optional): Config for testing. Default: None.
    """

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
        n_ranks=3,
        n_vertices=33,
        en_adaint=True,
        en_adaint_share=False,
        backbone='tpami',
        pretrained=False,
        n_colors=3,
        sparse_factor=0.0001,
        smooth_factor=0,
        monotonicity_factor=10.0,
        recons_loss=dict(type='L2Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None):

        super().__init__()

        assert backbone.lower() in ['tpami', 'res18']

        # mapping f
        self.backbone = dict(
            tpami=TPAMIBackbone,
            res18=Res18Backbone)[backbone.lower()](pretrained, extra_pooling=en_adaint)

        # mapping h
        self.lut_generator = LUTGenerator(
            n_colors, n_vertices, self.backbone.out_channels, n_ranks)

        # mapping g
        if en_adaint:
            self.adaint = AdaInt(
                n_colors, n_vertices, self.backbone.out_channels, en_adaint_share)
        else:
            uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1) \
                                    .repeat(n_colors, 1)
            self.register_buffer('uniform_vertices', uniform_vertices.unsqueeze(0))

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = en_adaint
        self.sparse_factor = sparse_factor
        self.smooth_factor = smooth_factor
        self.monotonicity_factor = monotonicity_factor
        self.backbone_name = backbone.lower()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False

        self.init_weights()

        self.recons_loss = build_loss(recons_loss)

        # fix AdaInt for some steps
        self.n_fix_iters = train_cfg.get('n_fix_iters', 0) if train_cfg else 0
        self.adaint_fixed = False
        self.register_buffer('cnt_iters', torch.zeros(1))

        self.inference_times = []

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        For the mapping g (`adaint`), we use all-zero and all-one initializations for its weights
        and bias, respectively.
        """
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        self.lut_generator.init_weights()
        if self.en_adaint:
            self.adaint.init_weights()

    def forward_dummy(self, imgs):
        r"""The real implementation of model forward.

        Args:
            img (Tensor): Input image, shape (b, c, h, w).
        Returns:
            tuple(Tensor, Tensor, Tensor):
                Output image, LUT weights, Sampling Coordinates.
        """
        start = time.perf_counter()
        # E: (b, f)
        codes = self.backbone(imgs)
        # (b, m), T: (b, c, d, d, d)
        weights, luts = self.lut_generator(codes)
        # \hat{P}: (b, c, d)
        if self.en_adaint:
            vertices = self.adaint(codes)
            outs = lookup(imgs, luts, vertices)
        else:
            vertices = self.uniform_vertices
            outs = lookup(imgs, luts, vertices.repeat([luts.shape[0],1,1]))
        # import ipdb; ipdb.set_trace()

        end = time.perf_counter()
        self.inference_times.append(end - start)

        return outs, weights, vertices

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        r"""Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor, optional): Ground-truth image. Default: None.
            test_mode (bool, optional): Whether in test mode or not. Default: False.
            kwargs (dict, optional): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        r"""Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            outputs (dict): Output results.
        """
        losses = dict()
        output, weights, vertices = self.forward_dummy(lq)
        losses['loss_recons'] = self.recons_loss(output, gt)
        if self.sparse_factor > 0:
            losses['loss_sparse'] = self.sparse_factor * torch.mean(weights.pow(2))
        reg_smoothness, reg_monotonicity = self.lut_generator.regularizations(
            self.smooth_factor, self.monotonicity_factor)
        if self.smooth_factor > 0:
            losses['loss_smooth'] = reg_smoothness
        if self.monotonicity_factor > 0:
            losses['loss_mono'] = reg_monotonicity
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        r"""Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor, optional): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool, optional): Whether to save image. Default: False.
            save_path (str, optional): Path to save image. Default: None.
            iteration (int, optional): Iteration for the saving image name.
                Default: None.
        Returns:
            outputs (dict): Output results.
        """
        output, _, _ = self.forward_dummy(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.jpg')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.jpg')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def train_step(self, data_batch, optimizer):
        r"""Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.
        Returns:
            dict: Returned output.
        """
        # fix AdaInt in the first several epochs
        if self.en_adaint and self.cnt_iters < self.n_fix_iters:
            if not self.adaint_fixed:
                self.adaint_fixed = True
                self.adaint.requires_grad_(False)
                get_root_logger().info(f'Fix AdaInt for {self.n_fix_iters} iters.')
        elif self.en_adaint and self.cnt_iters == self.n_fix_iters:
            self.adaint.requires_grad_(True)
            if self.adaint_fixed:
                self.adaint_fixed = False
                get_root_logger().info(f'Unfix AdaInt after {self.n_fix_iters} iters.')

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})

        self.cnt_iters += 1
        return outputs

    def val_step(self, data_batch, **kwargs):
        r"""Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict, optional): Other arguments for ``val_step``.
        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def evaluate(self, output, gt):
        r"""Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        gt = (gt * 255).to(torch.uint8) / 255.
        output = (torch.clip(output, 0, 1) * 255).to(torch.uint8) / 255.

        # output = tensor2img(output)
        # gt = tensor2img(gt)

        eval_result = dict()
        eval_result["PSNR"] = (20 * torch.log10(1 / torch.sqrt(torch.mean((output - gt) ** 2, axis=(1,2,3))))).mean().item()
        from ptcolor import rgb2lab
        from pytorch_msssim import ssim
        output_lab = rgb2lab(output)
        gt_lab = rgb2lab(gt)
        eval_result["delta_Eab"] = (((output_lab - gt_lab) ** 2).sum(dim=1)**0.5).mean(dim=(1,2)).mean().item()

        for i in range(output.shape[0]):
            assert i==0
            H, W = 512, 512
            down_ratio = max(1, round(min(H, W) / 256))
            eval_result["SSIM"] = ssim(F.adaptive_avg_pool2d(gt[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
                        F.adaptive_avg_pool2d(output[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
                        data_range=1, size_average=False).item()

        # for metric in self.test_cfg.metrics:
        #     eval_result[metric] = self.allowed_metrics[metric](
        #         output, gt, crop_border)
        return eval_result

class IA_NILUT(nn.Module):
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

    def forward(self, intensity):
        input = torch.cat([intensity, torch.sort(intensity[:,:,:3])[0]], axis=2)
        zero_input = input[:1].clone().detach()
        zero_input[:, :, 3:-3] = 0
        input = torch.cat([input, zero_input], axis=0)
        output = self.net(input)
        output = output[:-1] - output[-1:] + intensity[:,:,:3]
        return output

clip_model = None

@MODELS.register_module()
class PG_IA_NILUT_with_PARAMS(BaseModel):

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
        n_vertices=33,
        n_colors=3,
        pg_loss_factor=1,
        recons_loss=dict(type='L2Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None,
        classnames=None,
        loose_loss=False,
        detach=False,
        reverse_loss=False,
        without_psnr=False,
        target_loss_weight=None,
        pretrained_path=None,
        psnr_factor=1,
        pg_loss_freq=1,
        param_scale=1,
        dataset="fiveK"):

        super().__init__()
        self.n_ranks = len(classnames)

        if dataset == "fiveK":
            with open("data/FiveK/train.txt", 'r') as f:
                self.image_names = f.read().split("\n")[:-1]
        elif dataset == "ppr10K":
            with open("data/PPR10K/train.txt", 'r') as f:
                self.image_names = f.read().split("\n")[:-1]
        self.image_names = sorted(self.image_names)

        self.n_colors = n_colors
        self.n_vertices = n_vertices

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.recons_loss = build_loss(recons_loss)

        self.register_buffer('cnt_iters', torch.zeros(1))

        clip_cfg = dict(
            type='CLIPIQAFixed',
            backbone_name='RN50',
            classnames=classnames)
        global clip_model
        clip_model = build_backbone(clip_cfg).cuda()
        clip_model.requires_grad_(False)

        self.classnames = classnames

        self.uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1) \
                                .repeat(n_colors, 1).unsqueeze(0).cuda()
        self.identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.identity_lut = self.identity_lut[0].reshape((1, 3, -1)).cuda()

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

        self.pg_loss_target = torch.zeros((self.n_ranks*2, self.n_ranks)).cuda()
        for i in range(self.n_ranks):
            self.pg_loss_target[i*2, i] = 1
            self.pg_loss_target[i*2+1, i] = -1

        self.pg_loss_factor = pg_loss_factor
        self.epsilon = 10**-8
        self.without_psnr=without_psnr
        self.reverse_loss = reverse_loss
        self.psnr_factor = psnr_factor
        self.pg_loss_freq = pg_loss_freq
        self.param_scale = param_scale

        if target_loss_weight is not None:
            self.target_loss_weight = torch.tensor(target_loss_weight).cuda()
        else:
            self.target_loss_weight = None

        self.IA_NILUT = IA_NILUT(in_features=3+self.n_ranks)

        self.params = nn.Sequential(nn.Linear(len(self.image_names), self.n_ranks), nn.Tanh())
        nn.init.zeros_(self.params[0].bias)
        nn.init.zeros_(self.params[0].weight)

        if pretrained_path is not None:
            incompatibleKeys = self.load_state_dict(torch.load(pretrained_path)["state_dict"], strict=False)
            print(incompatibleKeys)

    def init_weights(self):
        pass

    def forward_dummy(self, imgs, image_id, imgs_hq=None):
        tiled_params = self.params(torch.eye(self.params[0].weight.shape[1])[image_id].cuda() * self.param_scale)[:,:,None].repeat([1,1,self.identity_lut.shape[2]])
        ins = torch.cat([self.identity_lut.repeat([tiled_params.shape[0],1,1]), tiled_params], axis=1)

        if not torch.is_grad_enabled():
            ins = torch.permute(ins, (0, 2, 1))
            output_luts = self.IA_NILUT(ins)
            output_luts = torch.permute(output_luts, (0, 2, 1))
            output_luts = output_luts.reshape((-1, 3, 33, 33, 33))
            outs = lookup(imgs.repeat([output_luts.shape[0]//imgs.shape[0],1,1,1]), output_luts, self.uniform_vertices.repeat([output_luts.shape[0],1,1]))
            return outs, None, None

        base_params = torch.nn.parameter.Parameter(torch.rand([imgs.shape[0], self.n_ranks]).cuda() * 2 * 1 - 1)
        base_tiled_params = base_params[:,:,None].repeat([1,1,self.identity_lut.shape[2]])
        base_ins = torch.cat([self.identity_lut.repeat([base_tiled_params.shape[0],1,1]), base_tiled_params], axis=1)
        ins = torch.cat([ins, base_ins], axis=0)

        if imgs_hq is not None:
            reverse_ins = torch.cat([self.identity_lut.repeat([tiled_params.shape[0],1,1]), -tiled_params], axis=1)
            ins = torch.cat([ins, reverse_ins], axis=0)

        ins = torch.permute(ins, (0, 2, 1))
        output_luts = self.IA_NILUT(ins)
        output_luts = torch.permute(output_luts, (0, 2, 1))
        output_luts = output_luts.reshape((-1, 3, 33, 33, 33))

        if imgs_hq is not None:
            output_reverse_luts = output_luts[-reverse_ins.shape[0]:]
            output_luts = output_luts[:-reverse_ins.shape[0]]

        outs = lookup(imgs.repeat([output_luts.shape[0]//imgs.shape[0],1,1,1]), output_luts, self.uniform_vertices.repeat([output_luts.shape[0],1,1]))
        if imgs_hq is not None:
            reverse_outs = lookup(imgs_hq.repeat([output_reverse_luts.shape[0]//imgs.shape[0],1,1,1]), output_reverse_luts, self.uniform_vertices.repeat([output_reverse_luts.shape[0],1,1]))
        else:
            reverse_outs = None

        outs_ = outs[:imgs.shape[0]]
        base_outs = outs[-imgs.shape[0]:]

        base_ = (base_outs - self.mean.reshape((1, 3, 1, 1))) / self.std.reshape((1, 3, 1, 1))
        if self.pg_loss_factor != 0 and self.cnt_iters % self.pg_loss_freq == 0:
            clip_score_base = clip_model(base_)[0]

            if torch.is_grad_enabled():
                from torch.autograd import grad
                gradients = torch.zeros((imgs.shape[0], self.n_ranks, self.n_ranks)).cuda()
                for i in range(self.n_ranks):
                    if base_params.grad is not None:
                        base_params.grad.data.zero_()
                    gradients[:,i] = grad(clip_score_base[:,i], base_params, grad_outputs=torch.ones(imgs.shape[0]).cuda(), retain_graph=True, create_graph=True)[0]
            else:
                gradients = None
        else:
            gradients = None
        return outs_, reverse_outs, gradients

    def pg_loss(self, gradients):
        pg_loss_target_ = torch.eye(self.n_ranks)[None].cuda()

        if self.target_loss_weight is not None:
            target_loss_weight_ = (self.target_loss_weight[:, None] * torch.abs(pg_loss_target_)).repeat_interleave(gradients.shape[0], dim=0)
            loss_1 = (torch.abs(pg_loss_target_ - gradients) * target_loss_weight_).sum() / torch.abs(pg_loss_target_).sum()
        else:
            loss_1 = (torch.abs(pg_loss_target_ - gradients) * torch.abs(pg_loss_target_)).sum() / torch.abs(pg_loss_target_).sum()
        loss_2 = (torch.abs(pg_loss_target_ - gradients) * (1 - torch.abs(pg_loss_target_))).sum() / (1 - torch.abs(pg_loss_target_)).sum()

        return (loss_1 * 0.2 + loss_2 * 0.8) * self.pg_loss_factor

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        r"""Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor, optional): Ground-truth image. Default: None.
            test_mode (bool, optional): Whether in test mode or not. Default: False.
            kwargs (dict, optional): Other arguments.
        """
        if test_mode:
            image_name = os.path.basename(kwargs["meta"][0]['lq_path'])[:-4]
            if image_name in self.image_names:
                image_ids = [self.image_names.index(image_name)]
            else:
                image_ids = [0]
            return self.forward_test(lq, gt, image_ids, **kwargs)

        image_ids = []
        for i in range(len(kwargs["meta"])):
            image_name = os.path.basename(kwargs["meta"][i]['lq_path'])[:-4]
            image_id = self.image_names.index(image_name)
            image_ids.append(image_id)
        return self.forward_train(lq, gt, image_ids)

    def forward_train(self, lq, gt, image_ids):
        losses = dict()
        if self.reverse_loss:
            output, reverse_output, gradients = self.forward_dummy(lq, image_ids, imgs_hq=gt)
        else:
            output, _, gradients = self.forward_dummy(lq, image_ids)

        if not self.without_psnr:
            losses['loss_recons'] = self.recons_loss(output, gt) * self.psnr_factor
        if gradients is not None:
            losses['loss_pg'] = self.pg_loss(gradients)
        if self.reverse_loss:
            if not self.without_psnr:
                losses['loss_recons_reverse'] = self.recons_loss(reverse_output, lq) * self.psnr_factor

        if self.without_psnr:
            outputs = dict(
                losses=losses,
                num_samples=len(gt.data),
                results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        else:
            psnr = (20 * torch.log10(1 / torch.sqrt(torch.mean((output - gt) ** 2, axis=(1,2,3))))).mean()
            outputs = dict(
                losses=losses,
                num_samples=len(gt.data),
                results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()),
                psnr=psnr)
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     image_ids=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        output, reverse_output, gradients = self.forward_dummy(lq, image_ids)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images。')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        import numpy as np
        if results['eval_result']["PSNR"] == np.inf:
            results['eval_result']["PSNR"] = (20 * torch.log10(1 / torch.sqrt(torch.mean((output - gt) ** 2, axis=(1,2,3))))).mean().item()

        return results

    def train_step(self, data_batch, optimizer):
        r"""Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.
        Returns:
            dict: Returned output.
        """

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not self.without_psnr:
            log_vars["psnr"] = outputs["psnr"].item()
        outputs.update({'log_vars': log_vars})

        self.cnt_iters += 1
        return outputs

    def val_step(self, data_batch, **kwargs):
        r"""Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict, optional): Other arguments for ``val_step``.
        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def evaluate(self, output, gt):
        r"""Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](
                output, gt, crop_border)
        return eval_result


@MODELS.register_module()
class PG_IA_NILUT_without_PARAMS(PG_IA_NILUT_with_PARAMS):
    def __init__(self,
        n_vertices=33,
        n_colors=3,
        pg_loss_factor=1,
        recons_loss=dict(type='L2Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None,
        classnames=None,
        loose_loss=False,
        detach=False,
        reverse_loss=False,
        target_loss_weight=None,
        pretrained_path=None,
        psnr_factor=1,
        pg_loss_freq=1,
        param_scale=1,
        dataset="fiveK"):

        super().__init__(
            n_vertices=n_vertices,
            n_colors=n_colors,
            pg_loss_factor=pg_loss_factor,
            recons_loss=recons_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            classnames=classnames,
            loose_loss=loose_loss,
            detach=detach,
            reverse_loss=reverse_loss,
            without_psnr=True,
            target_loss_weight=target_loss_weight,
            pretrained_path=pretrained_path,
            psnr_factor=psnr_factor,
            pg_loss_freq=pg_loss_freq,
            param_scale=param_scale,
            dataset=dataset)



@MODELS.register_module()
class PARAM_PREDICTOR(BaseModel):

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
        n_ranks=3,
        n_vertices=33,
        backbone='tpami',
        pretrained=False,
        n_colors=3,
        recons_loss=dict(type='L2Loss', loss_weight=1.0, reduction='mean'),
        train_cfg=None,
        test_cfg=None,
        pretrained_PG_IA_NILUT_path=None,):

        super().__init__()

        self.n_ranks = n_ranks

        assert backbone.lower() in ['tpami', 'res18']

        self.backbone = dict(
            tpami=TPAMIBackbone,
            res18=Res18Backbone)[backbone.lower()](pretrained, extra_pooling=True)

        self.weights_generator = WeightsGenerator(self.backbone.out_channels, n_ranks)

        self.PG_IA_NILUT = IA_NILUT(in_features=3+self.n_ranks)

        self.PG_IA_NILUT.requires_grad_(False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.backbone_name = backbone.lower()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained_PG_IA_NILUT_path = pretrained_PG_IA_NILUT_path

        self.fp16_enabled = False

        self.init_weights()

        self.recons_loss = build_loss(recons_loss)

        self.inference_times = []

        self.uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1) \
                                .repeat(n_colors, 1).unsqueeze(0).cuda()
        self.identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.identity_lut = self.identity_lut[0].reshape((1, 3, -1)).cuda()

        self.psnrs = []

        if "visualize" in self.test_cfg.keys():
            self.visualize = self.test_cfg["visualize"]
        else:
            self.visualize = False


    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        For the mapping g (`adaint`), we use all-zero and all-one initializations for its weights
        and bias, respectively.
        """
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)

        self.weights_generator.init_weights()

        PG_IA_NILUT_weights = {}
        if os.path.exists(self.pretrained_PG_IA_NILUT_path):
            PG_IA_NILUT_pretrained_weights = torch.load(self.pretrained_PG_IA_NILUT_path)['state_dict']
            for key in self.PG_IA_NILUT.state_dict().keys():
                PG_IA_NILUT_weights[key] = PG_IA_NILUT_pretrained_weights["IA_NILUT."+key]
            self.PG_IA_NILUT.load_state_dict(PG_IA_NILUT_weights)
        else:
            print(self.pretrained_PG_IA_NILUT_path, "does not exist.")

    def forward_dummy(self, imgs):
        start = time.perf_counter()
        codes = self.backbone(imgs)
        weights = self.weights_generator(codes)

        tiled_params = weights[:,:,None].repeat([1,1,self.identity_lut.shape[2]])
        ins = torch.cat([self.identity_lut.repeat([tiled_params.shape[0],1,1]), tiled_params], axis=1)
        ins = torch.permute(ins, (0, 2, 1))
        output_luts = self.PG_IA_NILUT(ins)
        output_luts = torch.permute(output_luts, (0, 2, 1))
        output_luts = output_luts.reshape((-1, 3, self.n_vertices, self.n_vertices, self.n_vertices))

        outs = lookup(imgs, output_luts, self.uniform_vertices.repeat([output_luts.shape[0],1,1]))

        end = time.perf_counter()
        self.inference_times.append(end - start)

        return outs, weights


    def forward_dummy_visualize(self, imgs, id):
        start = time.perf_counter()
        codes = self.backbone(imgs)
        weights = self.weights_generator(codes)

        tiled_params = weights[:,:,None].repeat([1,1,self.identity_lut.shape[2]])
        ins = torch.cat([self.identity_lut, tiled_params], axis=1)
        zero_tiled_params = torch.zeros_like(tiled_params)
        zero_tiled_params[:,self.n_ranks:] = tiled_params[:,self.n_ranks:]
        zero_ins = torch.cat([self.identity_lut, zero_tiled_params], axis=1)
        ins = torch.cat([ins, zero_ins], axis=0)

        for i in range(self.n_ranks):
            plus_tiled_params = torch.zeros_like(tiled_params)
            plus_tiled_params[:,self.n_ranks:] = tiled_params[:,self.n_ranks:]
            plus_tiled_params[:, i] = 1
            plus_inter_tiled_params = plus_tiled_params * 0.5
            minus_tiled_params = plus_tiled_params * -1
            minus_inter_tiled_params = plus_inter_tiled_params * -1
            current_tiled_params = tiled_params * 1.
            current_tiled_params[:, i+1:] = 0

            plus_ins = torch.cat([self.identity_lut, plus_tiled_params], axis=1)
            plus_inter_ins = torch.cat([self.identity_lut, plus_inter_tiled_params], axis=1)
            minus_ins = torch.cat([self.identity_lut, minus_tiled_params], axis=1)
            minus_inter_ins = torch.cat([self.identity_lut, minus_inter_tiled_params], axis=1)
            current_ins = torch.cat([self.identity_lut, current_tiled_params], axis=1)
            ins = torch.cat([ins, plus_ins, plus_inter_ins, minus_ins, minus_inter_ins, current_ins], axis=0)

        ins = torch.permute(ins, (0, 2, 1))
        output_luts = self.PG_IA_NILUT(ins)
        output_luts = torch.permute(output_luts, (0, 2, 1))
        output_luts = output_luts.reshape((-1, 3, self.n_vertices, self.n_vertices, self.n_vertices))

        outs = lookup(imgs.repeat([output_luts.shape[0],1,1,1]), output_luts, self.uniform_vertices.repeat([output_luts.shape[0],1,1]))

        outs_ = outs[:imgs.shape[0]]
        zero_outs = outs[imgs.shape[0]:imgs.shape[0]*2]
        plus_minus_outs = outs[imgs.shape[0]*2:]

        return outs_, zero_outs, plus_minus_outs


    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        losses = dict()
        output, weights = self.forward_dummy(lq)
        losses['loss_recons'] = self.recons_loss(output, gt)

        psnr = 20 * torch.log10(1 / torch.sqrt(torch.mean((output - gt) ** 2)))

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()),
            psnr=psnr)

        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):

        if save_image:
            if self.visualize:
                output, zero_output, plus_minus_output = self.forward_dummy_visualize(lq, os.path.basename(meta[0]['lq_path'])[:-4])

                for i in range(plus_minus_output.shape[0]//5):
                    outputs_row = torch.cat([plus_minus_output[i*5+2:i*5+3], plus_minus_output[i*5+3:i*5+4], zero_output, plus_minus_output[i*5+1:i*5+2], plus_minus_output[i*5:i*5+1]], axis=3)
                    if i == 0:
                        outputs = outputs_row
                    else:
                        margin = torch.ones_like(outputs_row)[:,:,:50]
                        outputs = torch.cat([outputs, margin, outputs_row], axis=2)
                lq_path = meta[0]['lq_path']
                folder_name = osp.splitext(osp.basename(lq_path))[0]

                save_path = osp.join(save_path, f'{folder_name}.jpg')
                outputs = F.interpolate(outputs, scale_factor=0.25)
                mmcv.imwrite(tensor2img(outputs), save_path)

            else:
                output, weights = self.forward_dummy(lq)

                lq_path = meta[0]['lq_path']
                folder_name = osp.splitext(osp.basename(lq_path))[0]
                if isinstance(iteration, numbers.Number):
                    save_path = osp.join(save_path, folder_name,
                                         f'{folder_name}-{iteration + 1:06d}.jpg')
                elif iteration is None:
                    save_path = osp.join(save_path, f'{folder_name}.jpg')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), save_path)
        else:
            output, weights = self.forward_dummy(lq)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()
        return results

    def train_step(self, data_batch, optimizer):
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_vars["psnr"] = outputs["psnr"].item()
        outputs.update({'log_vars': log_vars})

        return outputs

    def val_step(self, data_batch, **kwargs):
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def evaluate(self, output, gt):
        if output is None:
            return {}
        crop_border = self.test_cfg.crop_border
        gt = torch.clip(gt * 255, 0, 255).to(torch.uint8) / 255.
        output = torch.clip(output * 255, 0, 255).to(torch.uint8) / 255.

        eval_result = dict()
        eval_result["PSNR"] = (20 * torch.log10(1 / torch.sqrt(torch.mean((output - gt) ** 2, axis=(1,2,3))))).mean().item()
        from ptcolor import rgb2lab
        from pytorch_msssim import ssim
        output_lab = rgb2lab(output)
        gt_lab = rgb2lab(gt)
        eval_result["delta_Eab"] = (((output_lab - gt_lab) ** 2).sum(dim=1)**0.5).mean(dim=(1,2)).mean().item()

        for i in range(output.shape[0]):
            assert i==0
            H, W = 512, 512
            down_ratio = max(1, round(min(H, W) / 256))
            eval_result["SSIM"] = ssim(F.adaptive_avg_pool2d(gt[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
                        F.adaptive_avg_pool2d(output[i:i+1], (int(H / down_ratio), int(W / down_ratio))),
                        data_range=1, size_average=False).item()

        return eval_result
