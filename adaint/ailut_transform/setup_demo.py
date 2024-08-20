import os
import os.path as osp
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension #, CUDAExtension


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for extension in self.extensions:
            if isinstance(extension, dict):
                extension.extra_compile_args['cxx'] = ['-pthread']
            else:
                pass
                # extension.extra_compile_args = ['-pthread']
            extension.extra_link_args = ['-L/lib/x86_64-linux-gnu', '-L/usr/lib/x86_64-linux-gnu', '-lpthread']
        super().build_extensions()


def get_version(version_file):
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


os.chdir(osp.dirname(osp.abspath(__file__)))
csrc_directory = osp.join('ailut_demo', 'csrc')
setup(
    name='ailut_demo',
    version=get_version(osp.join('ailut_demo', 'version.py')),
    description='Adaptive Interval 3D LookUp Table Transform',
    author='Charles',
    author_email='charles.young@sjtu.edu.cn',
    packages=find_packages(),
    include_package_data=False,
    ext_modules=[
        # CUDAExtension('ailut_demo._ext', [
        CppExtension('ailut_demo._ext', [
            osp.join(csrc_directory, 'ailut_transform.cpp'),
            osp.join(csrc_directory, 'ailut_transform_cpu.cpp'),
            # osp.join(csrc_directory, 'ailut_transform_cuda.cu')
        ])
    ],
    cmdclass={
        'build_ext': CustomBuildExtension # BuildExtension
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    license='Apache License 2.0',
    zip_safe=False
)
