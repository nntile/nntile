# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/setup.py
# Build thin PyTorch PrivateUse1 extension linking prebuilt libtorch_nntile.

import os
import subprocess
import sys
from pathlib import Path

# Cloud images may default to clang without libstdc++ headers; g++ is required.
os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")

try:
    import torch
except ImportError as exc:  # pragma: no cover - build-time guard
    raise SystemExit(
        "torch_nntile requires PyTorch to build the native extension.\n"
        "Install the pinned ABI first, then retry:\n"
        "  pip install 'torch==2.9.1' 'torchvision==0.24.1'\n"
        "Or use a matching CPU/CUDA wheel index for your platform."
    ) from exc

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

# Bridge / models live in shared libtorch_nntile; _C is pybind only.
EXT_SOURCES = ["csrc/nntile_module.cpp"]


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


def _lib_candidates(directory: Path, basenames: list[str]) -> bool:
    if not directory.is_dir():
        return False
    for base in basenames:
        for pattern in (f"lib{base}.so", f"lib{base}.dylib", f"{base}.lib"):
            if (directory / pattern).exists():
                return True
            # libfoo.so.1 style
            if any(directory.glob(f"lib{base}.so*")):
                return True
    return False


def _resolve_lib_layout() -> tuple[Path, Path, Path, Path]:
    """Return (nntile_lib_dir, torch_nntile_lib_dir, nntile_include, nntile_source).

    Prefer cmake install prefixes (NNTILE_PREFIX / TORCH_NNTILE_PREFIX).
    Fall back to build trees (NNTILE_BUILD_DIR / TORCH_NNTILE_BUILD_DIR).
    """
    ci_build_wheel = os.environ.get("CIBUILDWHEEL") == "1"

    if (var := os.environ.get("NNTILE_SOURCE_DIR")):
        nntile_source = Path(var)
    else:
        nntile_source = REPO_ROOT

    nntile_prefix = os.environ.get("NNTILE_PREFIX")
    torch_prefix = os.environ.get("TORCH_NNTILE_PREFIX", nntile_prefix)

    if nntile_prefix:
        nntile_pref = Path(nntile_prefix)
        torch_pref = Path(torch_prefix) if torch_prefix else nntile_pref
        nntile_lib = nntile_pref / "lib"
        torch_lib = torch_pref / "lib"
        nntile_inc = nntile_pref / "include"
        if not _lib_candidates(nntile_lib, ["nntile"]):
            raise RuntimeError(
                f"NNTILE_PREFIX={nntile_pref!r} has no libnntile under {nntile_lib}"
            )
        if not _lib_candidates(torch_lib, ["torch_nntile"]):
            raise RuntimeError(
                f"TORCH_NNTILE_PREFIX={torch_pref!r} has no "
                f"libtorch_nntile under {torch_lib}"
            )
        if not (nntile_inc / "nntile" / "defs.h").exists() and not (
            nntile_inc / "nntile.hh"
        ).exists():
            raise RuntimeError(
                f"NNTILE_PREFIX={nntile_pref!r} missing installed headers "
                f"under {nntile_inc}"
            )
        return nntile_lib, torch_lib, nntile_inc, nntile_source

    if (var := os.environ.get("NNTILE_BUILD_DIR")):
        nntile_build = Path(var)
    elif ci_build_wheel:
        nntile_build = nntile_source / "build" / "torch_nntile_wheel"
    else:
        raise RuntimeError(
            "torch_nntile requires prebuilt libnntile and libtorch_nntile.\n"
            "Install both (cmake --install) and set:\n"
            "  export NNTILE_PREFIX=$PWD/install\n"
            "  export TORCH_NNTILE_PREFIX=$PWD/install   # if same prefix\n"
            "Or point at CMake build trees:\n"
            "  export NNTILE_BUILD_DIR=$PWD/build\n"
            "  export TORCH_NNTILE_BUILD_DIR=$PWD/build   # contains torch_nntile/\n"
            "  export NNTILE_SOURCE_DIR=$PWD\n"
            "  CXX=g++ pip install -e ./torch_nntile --no-build-isolation"
        )

    if (var := os.environ.get("TORCH_NNTILE_BUILD_DIR")):
        torch_build = Path(var)
    else:
        torch_build = nntile_build

    nntile_lib = nntile_build / "nntile"
    torch_lib = torch_build / "torch_nntile"
    nntile_inc = nntile_build / "include"
    if not _lib_candidates(nntile_lib, ["nntile"]):
        raise RuntimeError(
            f"NNTILE_BUILD_DIR={nntile_build!r} does not contain "
            f"libnntile under {nntile_lib}"
        )
    if not _lib_candidates(torch_lib, ["torch_nntile"]):
        raise RuntimeError(
            f"TORCH_NNTILE_BUILD_DIR={torch_build!r} does not contain "
            f"libtorch_nntile under {torch_lib}. Build with "
            "-DBUILD_TORCH_NNTILE=ON first."
        )
    if not (nntile_inc / "nntile" / "defs.h").exists():
        raise RuntimeError(
            f"NNTILE_BUILD_DIR={nntile_build!r} missing generated "
            f"headers ({nntile_inc / 'nntile' / 'defs.h'})"
        )
    return nntile_lib, torch_lib, nntile_inc, nntile_source


def _nntile_extension_kwargs() -> dict:
    """Build kwargs for torch_nntile._C (links libtorch_nntile + libnntile)."""
    nntile_lib_dir, torch_lib_dir, nntile_inc, nntile_source = (
        _resolve_lib_layout()
    )

    cxx_standard = os.environ.get("TORCH_NNTILE_CXX_STANDARD", "c++17")
    extra_compile_args = [f"-std={cxx_standard}"]
    define_macros: list[tuple[str, str | None]] = [
        ("TORCH_NNTILE_USE_LIBNNTILE", "1"),
    ]
    # Public + private headers from this package source; nntile from prefix/build.
    include_dirs: list[str] = [
        str(ROOT / "include"),
        str(ROOT / "csrc"),
        str(nntile_inc),
        str(nntile_source / "nntile" / "include"),
    ]
    library_dirs: list[str] = [str(nntile_lib_dir), str(torch_lib_dir)]
    libraries: list[str] = ["torch_nntile", "nntile"]
    extra_link_args: list[str] = []
    if sys.platform == "darwin":
        extra_link_args.append("-Wl,-rpath,@loader_path/../torch/lib")

    if os.environ.get("TORCH_NNTILE_WHEEL") != "1":
        if sys.platform == "darwin":
            extra_link_args.append(f"-Wl,-rpath,{nntile_lib_dir}")
            extra_link_args.append(f"-Wl,-rpath,{torch_lib_dir}")
        else:
            extra_link_args.append(f"-Wl,-rpath,{nntile_lib_dir}")
            extra_link_args.append(f"-Wl,-rpath,{torch_lib_dir}")

    _apply_pkg_config(
        "starpu-1.4",
        include_dirs,
        library_dirs,
        extra_compile_args,
        extra_link_args,
        libraries,
    )
    include_dirs.extend(_cuda_include_dirs())

    return {
        "define_macros": define_macros,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "libraries": libraries,
        "extra_compile_args": extra_compile_args,
        "extra_link_args": extra_link_args,
    }


ext_kwargs = _nntile_extension_kwargs()

_wheel_version = os.environ.get("TORCH_NNTILE_WHEEL_VERSION", "0.0.5")
_torch_requires = "torch==2.9.1"
_torchvision_requires = "torchvision==0.24.1"
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
            sources=EXT_SOURCES,
            **ext_kwargs,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        _torch_requires,
        _torchvision_requires,
        *_linux_nvidia_requires,
    ],
)
