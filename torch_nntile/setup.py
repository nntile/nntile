# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/setup.py
# Build PyTorch PrivateUse1 extension for the nntile device.

import os
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

CSRC = [
    "csrc/nntile_allocator.cpp",
    "csrc/nntile_kernels.cpp",
    "csrc/nntile_guard.cpp",
    "csrc/nntile_generator.cpp",
    "csrc/nntile_hooks.cpp",
    "csrc/nntile_module.cpp",
    "csrc/nntile_context.cpp",
    "csrc/nntile_graph_recorder.cpp",
    "csrc/nntile_executor.cpp",
    "csrc/nntile_add.cpp",
    "csrc/nntile_mul.cpp",
    "csrc/nntile_linear.cpp",
    "csrc/nntile_relu.cpp",
    "csrc/nntile_threshold_backward.cpp",
    "csrc/nntile_silu.cpp",
    "csrc/nntile_silu_backward.cpp",
    "csrc/nntile_gelu.cpp",
    "csrc/nntile_gelu_backward.cpp",
    "csrc/nntile_mm.cpp",
    "csrc/nntile_cross_entropy.cpp",
    "csrc/nntile_sgd_step.cpp",
    "csrc/nntile_adam_step.cpp",
    "csrc/nntile_layer_norm.cpp",
    "csrc/nntile_rms_norm.cpp",
]


def _pkg_config(package: str, flag: str) -> list[str]:
    env = os.environ.copy()
    pkg_config_path = env.get("PKG_CONFIG_PATH", "")
    starpu_roots = []
    if starpu_prefix := os.environ.get("STARPU_PREFIX"):
        starpu_roots.append(f"{starpu_prefix}/lib/pkgconfig")
    starpu_roots.append("/opt/starpu/lib/pkgconfig")
    for starpu_pkg in starpu_roots:
        if starpu_pkg not in pkg_config_path.split(":"):
            pkg_config_path = (
                f"{starpu_pkg}:{pkg_config_path}" if pkg_config_path else starpu_pkg
            )
    env["PKG_CONFIG_PATH"] = pkg_config_path
    try:
        out = subprocess.check_output(
            ["pkg-config", flag, package],
            env=env,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [part for part in out.strip().split() if part]


def _apply_pkg_config(
    package: str,
    include_dirs: list[str],
    library_dirs: list[str],
    extra_compile_args: list[str],
    extra_link_args: list[str],
    libraries: list[str],
) -> None:
    for flag in _pkg_config(package, "--cflags"):
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])
        else:
            extra_compile_args.append(flag)
    for flag in _pkg_config(package, "--libs"):
        if flag.startswith("-L"):
            library_dirs.append(flag[2:])
        elif flag.startswith("-l"):
            libraries.append(flag[2:])
        else:
            extra_link_args.append(flag)


def _cudnn_include_dir() -> str | None:
    if os.environ.get("TORCH_NNTILE_USE_CUDA") != "1":
        return None
    if path := os.environ.get("CUDNN_INCLUDE_PATH"):
        return path
    try:
        import importlib

        mod = importlib.import_module("nvidia.cudnn")
        if getattr(mod, "__file__", None):
            root = Path(mod.__file__).resolve().parent
        else:
            root = Path(next(iter(mod.__path__)))
        include = root / "include"
        if (include / "cudnn.h").exists():
            return str(include)
    except ImportError:
        return None
    return None


def _cuda_include_dirs() -> list[str]:
    if os.environ.get("TORCH_NNTILE_USE_CUDA") != "1":
        return []
    dirs: list[str] = []
    if cuda_home := os.environ.get("CUDA_HOME"):
        cuda_include = Path(cuda_home) / "include"
        if cuda_include.is_dir():
            dirs.append(str(cuda_include))
    if cudnn_include := _cudnn_include_dir():
        dirs.append(cudnn_include)
    return dirs


def _nntile_extension_kwargs() -> dict:
    ci_build_wheel = os.environ.get("CIBUILDWHEEL") == "1"

    if (var := os.environ.get("NNTILE_SOURCE_DIR")):
        nntile_source = Path(var)
    else:
        nntile_source = REPO_ROOT

    if (var := os.environ.get("NNTILE_BUILD_DIR")):
        nntile_build = Path(var)
    elif ci_build_wheel:
        nntile_build = nntile_source / "build" / "torch_nntile_wheel"
    else:
        nntile_build = None

    require_libnntile = os.environ.get("TORCH_NNTILE_REQUIRE_LIBNNTILE") == "1"
    if ci_build_wheel:
        require_libnntile = True

    cxx_standard = os.environ.get("TORCH_NNTILE_CXX_STANDARD", "c++17")
    extra_compile_args = [f"-std={cxx_standard}"]
    define_macros: list[tuple[str, str | None]] = []
    include_dirs: list[str] = []
    library_dirs: list[str] = []
    libraries: list[str] = []
    extra_link_args: list[str] = []
    if sys.platform == "darwin":
        extra_link_args.append("-Wl,-rpath,@loader_path/../torch/lib")

    if nntile_build is not None:
        nntile_lib_dir = nntile_build / "nntile"
        nntile_header_dir = nntile_build / "include" / "nntile" / "defs.h"
        if require_libnntile and not nntile_lib_dir.exists():
            raise RuntimeError(
                f"NNTILE_BUILD_DIR={nntile_build!r} does not contain "
                "the expected nntile library directory"
            )
        if require_libnntile and not nntile_header_dir.exists():
            raise RuntimeError(
                f"NNTILE_BUILD_DIR={nntile_build!r} does not contain "
                "generated nntile headers"
            )
        define_macros.append(("TORCH_NNTILE_USE_LIBNNTILE", "1"))
        include_dirs.extend([
            str(nntile_source / "nntile" / "include"),
            str(nntile_build / "include"),
        ])
        library_dirs.append(str(nntile_lib_dir))
        libraries.append("nntile")
        if os.environ.get("TORCH_NNTILE_WHEEL") != "1":
            if sys.platform == "darwin":
                extra_link_args.append(
                    "-Wl,-rpath,@loader_path/../../build/nntile"
                )
            else:
                extra_link_args.append(
                    "-Wl,-rpath,$ORIGIN/../../build/nntile"
                )
        _apply_pkg_config(
            "starpu-1.4",
            include_dirs,
            library_dirs,
            extra_compile_args,
            extra_link_args,
            libraries,
        )
        include_dirs.extend(_cuda_include_dirs())
    elif require_libnntile:
        raise RuntimeError(
            "torch_nntile wheel builds require libnntile; set NNTILE_BUILD_DIR"
        )

    return {
        "define_macros": define_macros,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
    }


ext_kwargs = _nntile_extension_kwargs()

_wheel_version = os.environ.get("TORCH_NNTILE_WHEEL_VERSION", "0.0.1")
_torch_requires = "torch==2.9.1"
_linux_marker = 'platform_system == "Linux" and platform_machine == "x86_64"'
_linux_nvidia_requires = [
    f"nvidia-cublas-cu12>=12.8.4.1; {_linux_marker}",
    f"nvidia-cudnn-cu12>=9.10.2.21; {_linux_marker}",
    f"nvidia-cusparse-cu12>=12.5.8.93; {_linux_marker}",
    f"nvidia-cusolver-cu12>=11.7.3.90; {_linux_marker}",
    f"nvidia-nvjitlink-cu12>=12.8.93; {_linux_marker}",
    f"nvidia-cuda-runtime-cu12>=12.8.90; {_linux_marker}",
]

setup(
    name="torch_nntile",
    version=_wheel_version,
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="torch_nntile._C",
            sources=CSRC,
            **ext_kwargs,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[_torch_requires, *_linux_nvidia_requires],
)
