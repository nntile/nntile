#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tools/watch_gpu_peak.py
# Sidecar: poll nvidia-smi until a PID exits, write peak GiB.

"""Poll GPU memory.used until ``pid`` exits; write peak GiB to a file.

The train process must not call ``nvidia-smi`` (it stalls the train wall).
The ladder runner starts this sidecar against the child PID.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        stat = (proc / "stat").read_text().split()
    except OSError:
        return False
    # Field 3 is state; 'Z' is zombie (reaped next wait).
    return len(stat) >= 3 and stat[2] != "Z"


def _smi_mib(gpu: str) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print(
            "usage: watch_gpu_peak.py PID GPU_INDEX OUT_GIB",
            file=sys.stderr,
        )
        return 2
    pid = int(args[0])
    gpu = args[1]
    out = Path(args[2])
    peak_mib = 0.0
    while _pid_alive(pid):
        sample = _smi_mib(gpu)
        if sample is not None and sample > peak_mib:
            peak_mib = sample
        time.sleep(0.05)
    sample = _smi_mib(gpu)
    if sample is not None and sample > peak_mib:
        peak_mib = sample
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{peak_mib / 1024.0:.4f}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
