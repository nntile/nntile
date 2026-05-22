# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file wrappers/python/nntile/graph_capture_sched/__init__.py
# Public entry point for StarPU **graph batch capture**
# (recording_begin / recording_end).
# Works with the SGOC loadable policy (``STARPU_SCHED=sgoc`` and
# ``STARPU_SCHED_LIB`` pointing at ``libgraph_sgoc_sched.so`` from starpu-sched
# ``new_sched``).
# The DSO exports ``starpu_graph_sched_graph_recording_begin`` / ``end``.
# For other schedulers (e.g. dmdasd), calls are no-ops so training can always
# bracket batches.
#
# @version 1.1.0

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path
from typing import Optional

__all__ = (
    "EXPECTED_SGOC_SCHEDULER_NAME",
    "EXPECTED_SGOC_SCHED_LIB_BASENAME",
    "GRAPH_CAPTURE_SCHED_ACTIVE",
    "graph_recording_begin",
    "graph_recording_end",
)

EXPECTED_SGOC_SCHEDULER_NAME = "sgoc"
EXPECTED_SGOC_SCHED_LIB_BASENAME = "libgraph_sgoc_sched.so"


def _sched_lib_path_matches_sgoc(path: str) -> bool:
    name = Path(path).name
    return name == EXPECTED_SGOC_SCHED_LIB_BASENAME or name.startswith(
        f"{EXPECTED_SGOC_SCHED_LIB_BASENAME}."
    )


def _env_matches_sgoc_scheduler_dso() -> bool:
    """True when env points at the SGOC scheduler DSO for recording."""
    sched = os.environ.get("STARPU_SCHED", "").strip()
    lib = os.environ.get("STARPU_SCHED_LIB", "").strip()
    if not lib:
        return False
    resolved = str(Path(lib).resolve())
    matches = _sched_lib_path_matches_sgoc(resolved)
    return sched == EXPECTED_SGOC_SCHEDULER_NAME and matches


# True when graph_recording_begin/end will call into STARPU_SCHED_LIB (SGOC).
GRAPH_CAPTURE_SCHED_ACTIVE = _env_matches_sgoc_scheduler_dso()

_scheduler_lib: Optional[ctypes.CDLL] = None


def _load_scheduler_lib() -> ctypes.CDLL:
    global _scheduler_lib
    if _scheduler_lib is not None:
        return _scheduler_lib
    if not _env_matches_sgoc_scheduler_dso():
        raise RuntimeError(
            "graph batch capture via scheduler DSO requires "
            f"STARPU_SCHED={EXPECTED_SGOC_SCHEDULER_NAME!r} with "
            f"STARPU_SCHED_LIB=.../{EXPECTED_SGOC_SCHED_LIB_BASENAME}"
        )
    path = Path(os.environ["STARPU_SCHED_LIB"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STARPU_SCHED_LIB is not a file: {path!r}")
    mode = 0
    for name in ("RTLD_NOW", "RTLD_GLOBAL"):
        mode |= getattr(ctypes, name, 0)
    try:
        _scheduler_lib = (
            ctypes.CDLL(path, mode=mode) if mode else ctypes.CDLL(path)
        )
    except OSError as e:
        msg = f"failed to load STARPU_SCHED_LIB {path!r}: {e}"
        raise RuntimeError(msg) from e
    lib = _scheduler_lib
    lib.starpu_graph_sched_graph_recording_begin.argtypes = (ctypes.c_uint,)
    lib.starpu_graph_sched_graph_recording_begin.restype = None
    lib.starpu_graph_sched_graph_recording_end.argtypes = (ctypes.c_uint,)
    lib.starpu_graph_sched_graph_recording_end.restype = None
    return lib


def graph_recording_begin(sched_ctx_id: int = 0) -> None:
    """Call ``starpu_graph_sched_graph_recording_begin`` from the scheduler
    DSO.

    ``sched_ctx_id=0`` uses the current StarPU scheduling context.
    No-op if ``STARPU_SCHED`` / ``STARPU_SCHED_LIB`` are not the SGOC pair.
    """
    if not _env_matches_sgoc_scheduler_dso():
        return
    lib = _load_scheduler_lib()
    if sched_ctx_id < 0:
        raise ValueError("sched_ctx_id must be non-negative")
    lib.starpu_graph_sched_graph_recording_begin(ctypes.c_uint(int(sched_ctx_id)))


def graph_recording_end(sched_ctx_id: int = 0) -> None:
    """Call ``starpu_graph_sched_graph_recording_end`` from the scheduler DSO.

    No-op if the active scheduler is not SGOC with a matching DSO path.
    """
    if not _env_matches_sgoc_scheduler_dso():
        return
    lib = _load_scheduler_lib()
    if sched_ctx_id < 0:
        raise ValueError("sched_ctx_id must be non-negative")
    lib.starpu_graph_sched_graph_recording_end(ctypes.c_uint(int(sched_ctx_id)))


if not GRAPH_CAPTURE_SCHED_ACTIVE:
    _sched = os.environ.get("STARPU_SCHED")
    _lib = os.environ.get("STARPU_SCHED_LIB")
    if _sched == EXPECTED_SGOC_SCHEDULER_NAME and _lib:
        lib_abs = str(Path(_lib).resolve())
        if not _sched_lib_path_matches_sgoc(lib_abs):
            warnings.warn(
                f"STARPU_SCHED is {EXPECTED_SGOC_SCHEDULER_NAME!r} but "
                f"STARPU_SCHED_LIB is not "
                f"{EXPECTED_SGOC_SCHED_LIB_BASENAME}*: {_lib!r}; "
                "nntile.graph_capture_sched.GRAPH_CAPTURE_SCHED_ACTIVE is "
                "False",
                RuntimeWarning,
                stacklevel=2,
            )
    elif _lib and _sched_lib_path_matches_sgoc(str(Path(_lib).resolve())):
        warnings.warn(
            f"STARPU_SCHED_LIB looks like SGOC ({_lib!r}) but STARPU_SCHED "
            f"is {_sched!r}, not {EXPECTED_SGOC_SCHEDULER_NAME!r}; "
            "nntile.graph_capture_sched.GRAPH_CAPTURE_SCHED_ACTIVE is False",
            RuntimeWarning,
            stacklevel=2,
        )
