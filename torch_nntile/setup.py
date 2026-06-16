# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/setup.py
# Build PyTorch PrivateUse1 extension for the nntile device stub.

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

CSRC = [
    "csrc/nntile_allocator.cpp",
    "csrc/nntile_kernels.cpp",
    "csrc/nntile_guard.cpp",
    "csrc/nntile_module.cpp",
]

setup(
    name="torch_nntile",
    version="0.1.0",
    packages=["torch_nntile"],
    ext_modules=[
        CppExtension(
            name="torch_nntile._C",
            sources=CSRC,
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=["torch>=2.1"],
)
