# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file wrappers/python/nntile/graph_recorder_sched/__init__.py
# Helpers when using the loadable graph_recorder StarPU scheduler from
# starpu-sched/new_sched (libgraph_recorder_sched.so): recording begin/end.
# When STARPU_SCHED is not graph_recorder (e.g. dmdasd), calls are no-ops so
# training pipelines can always bracket batches with begin/end.
#
# @version 1.0.0

from __future__ import annotations

import ctypes
import os
import warnings
from pathlib import Path
from typing import Optional

__all__ = (
    "GRAPH_RECORDER_ACTIVE",
    "EXPECTED_SCHEDULER_NAME",
    "EXPECTED_SCHED_LIB_BASENAMES",
    "graph_recording_begin",
    "graph_recording_end",
)

EXPECTED_SCHEDULER_NAME = "graph_recorder"
EXPECTED_SCHED_LIB_BASENAMES = (
    "libgraph_sched.so",
    "libgraph_recorder_sched.so",
)


def _sched_lib_path_matches(path: str) -> bool:
    name = Path(path).name
    if name in EXPECTED_SCHED_LIB_BASENAMES:
        return True
    return name.startswith("libgraph_sched.so.") or name.startswith(
        "libgraph_recorder_sched.so."
    )


def _env_matches_graph_recorder() -> bool:
    sched = os.environ.get("STARPU_SCHED", "")
    lib = os.environ.get("STARPU_SCHED_LIB", "")
    if sched != EXPECTED_SCHEDULER_NAME:
        return False
    if not lib.strip():
        return False
    resolved = os.path.abspath(lib)
    return _sched_lib_path_matches(resolved)


GRAPH_RECORDER_ACTIVE = _env_matches_graph_recorder()

_scheduler_lib: Optional[ctypes.CDLL] = None


def _load_scheduler_lib() -> ctypes.CDLL:
    global _scheduler_lib
    if _scheduler_lib is not None:
        return _scheduler_lib
    if not _env_matches_graph_recorder():
        raise RuntimeError(
            "graph_recorder recording requires "
            f"STARPU_SCHED={EXPECTED_SCHEDULER_NAME!r} and STARPU_SCHED_LIB "
            "pointing at libgraph_recorder_sched.so (or libgraph_sched.so)"
        )
    path = os.path.abspath(os.environ["STARPU_SCHED_LIB"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STARPU_SCHED_LIB is not a file: {path!r}")
    mode = 0
    for name in ("RTLD_NOW", "RTLD_GLOBAL"):
        mode |= getattr(ctypes, name, 0)
    try:
        _scheduler_lib = ctypes.CDLL(path, mode=mode) if mode else ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(f"failed to load STARPU_SCHED_LIB {path!r}: {e}") from e
    lib = _scheduler_lib
    lib.starpu_graph_sched_graph_recording_begin.argtypes = (ctypes.c_uint,)
    lib.starpu_graph_sched_graph_recording_begin.restype = None
    lib.starpu_graph_sched_graph_recording_end.argtypes = (ctypes.c_uint,)
    lib.starpu_graph_sched_graph_recording_end.restype = None
    return lib


def graph_recording_begin(sched_ctx_id: int = 0) -> None:
    """Call ``starpu_graph_sched_graph_recording_begin`` from the scheduler DSO.

    ``sched_ctx_id=0`` uses the current StarPU scheduling context.
    No-op if ``STARPU_SCHED`` is not ``graph_recorder`` (other schedulers).
    """
    if not _env_matches_graph_recorder():
        return
    lib = _load_scheduler_lib()
    if sched_ctx_id < 0:
        raise ValueError("sched_ctx_id must be non-negative")
    lib.starpu_graph_sched_graph_recording_begin(ctypes.c_uint(int(sched_ctx_id)))


def graph_recording_end(sched_ctx_id: int = 0) -> None:
    """Call ``starpu_graph_sched_graph_recording_end`` from the scheduler DSO.

    No-op if ``STARPU_SCHED`` is not ``graph_recorder``.
    """
    if not _env_matches_graph_recorder():
        return
    lib = _load_scheduler_lib()
    if sched_ctx_id < 0:
        raise ValueError("sched_ctx_id must be non-negative")
    lib.starpu_graph_sched_graph_recording_end(ctypes.c_uint(int(sched_ctx_id)))


if not GRAPH_RECORDER_ACTIVE:
    _sched = os.environ.get("STARPU_SCHED")
    _lib = os.environ.get("STARPU_SCHED_LIB")
    if _sched == EXPECTED_SCHEDULER_NAME and _lib:
        if not _sched_lib_path_matches(os.path.abspath(_lib)):
            warnings.warn(
                f"STARPU_SCHED is {EXPECTED_SCHEDULER_NAME!r} but STARPU_SCHED_LIB "
                f"basename is not a graph recorder DSO*: {_lib!r}; "
                "nntile.graph_recorder_sched.GRAPH_RECORDER_ACTIVE is False",
                RuntimeWarning,
                stacklevel=2,
            )
    elif _lib and _sched_lib_path_matches(os.path.abspath(_lib)):
        warnings.warn(
            f"STARPU_SCHED_LIB looks like graph_recorder ({_lib!r}) but "
            f"STARPU_SCHED is {_sched!r}, not {EXPECTED_SCHEDULER_NAME!r}; "
            "nntile.graph_recorder_sched.GRAPH_RECORDER_ACTIVE is False",
            RuntimeWarning,
            stacklevel=2,
        )
