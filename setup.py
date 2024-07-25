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
            ['csrc/pybind.cpp',
            'csrc/softmax_kernels.cu']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
