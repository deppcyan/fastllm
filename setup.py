# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11
import torch

setup(
    name='fastllm',
    ext_modules=[
        CUDAExtension(
            'fastllm', 
            [
            'csrc/pybind.cpp',
            'csrc/softmax_kernels.cu',
            'csrc/layers/mlp.cpp',
            'csrc/layers/attention.cpp',
            'csrc/layers/decoder_layer.cpp'
            ],
            extra_compile_args=['-g']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
