# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/setup.py
# Build PyTorch PrivateUse1 extension for the nntile device.

import os
import subprocess
from pathlib import Path

from setuptools import setup
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
    "csrc/nntile_executor.cpp",
    "csrc/nntile_add.cpp",
    "csrc/nntile_linear.cpp",
    "csrc/nntile_relu.cpp",
    "csrc/nntile_threshold_backward.cpp",
    "csrc/nntile_mm.cpp",
]


def _pkg_config(package: str, flag: str) -> list[str]:
    env = os.environ.copy()
    pkg_config_path = env.get("PKG_CONFIG_PATH", "")
    starpu_pkg = "/opt/starpu/lib/pkgconfig"
    if starpu_pkg not in pkg_config_path.split(":"):
        env["PKG_CONFIG_PATH"] = (
            f"{starpu_pkg}:{pkg_config_path}" if pkg_config_path else starpu_pkg
        )
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


def _nntile_extension_kwargs() -> dict:
    nntile_build = os.environ.get("NNTILE_BUILD_DIR")
    nntile_source = os.environ.get("NNTILE_SOURCE_DIR", str(REPO_ROOT))

    extra_compile_args = ["-std=c++17"]
    define_macros: list[tuple[str, str | None]] = []
    include_dirs: list[str] = []
    library_dirs: list[str] = []
    libraries: list[str] = []
    extra_link_args: list[str] = []

    if nntile_build:
        define_macros.append(("TORCH_NNTILE_USE_LIBNNTILE", "1"))
        include_dirs.extend(
            [
                str(Path(nntile_source) / "nntile" / "include"),
                str(Path(nntile_build) / "include"),
            ]
        )
        library_dirs.append(str(Path(nntile_build) / "nntile"))
        libraries.append("nntile")
        extra_link_args.append(
            f"-Wl,-rpath,$ORIGIN/../../build/nntile"
        )
        _apply_pkg_config(
            "starpu-1.4",
            include_dirs,
            library_dirs,
            extra_compile_args,
            extra_link_args,
            libraries,
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

setup(
    name="torch_nntile",
    version="0.5.0",
    packages=["torch_nntile"],
    ext_modules=[
        CppExtension(
            name="torch_nntile._C",
            sources=CSRC,
            **ext_kwargs,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=["torch>=2.1"],
)
