#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
# @file tools/chunked_coverage.py
# Inventory of the NNHaul-backend dispatch path per kernel op:
#   chunked  - a dedicated kernel::<op>::chunked entry point exists;
#   driver   - an nnhaul wrapper exists for the op (a chunk-aware driver or
#              a dispatch to a shared chunk kernel). This classifies by file
#              presence, not by inspecting the dispatch, so `classic` means
#              "no wrapper exists", not "a whole-buffer launch was found";
#   unused   - no core/tile/tensor wrapper reaches it (dead kernel);
#   classic  - no nnhaul wrapper and reachable: a default-product blocker to review.
# @version 1.1.0
"""Generate the chunked-coverage table for the NNHaul backend."""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNELS = ROOT / "nntile" / "include" / "nntile" / "kernel"
DRIVERS = ROOT / "nntile" / "src" / "nnhaul"


def chunked_ops():
    out = set()
    for hdr in KERNELS.glob("*/cuda.hh"):
        if re.search(r"\bchunked\s*\(", hdr.read_text(errors="ignore")):
            out.add(hdr.parent.name)
    return out


def driver_ops():
    """Wrappers that dispatch the op on the NNHaul backend."""
    skip = {"chunk_gate", "stage", "sources", "sync_defer"}
    return {src.stem for src in DRIVERS.glob("*.cc")} - skip


def used_by_high_level(op):
    """True when a core/tile/tensor wrapper reaches this kernel."""
    name = op.encode()
    for sub in ("core", "tile", "tensor"):
        for src in (ROOT / "nntile" / "src" / sub).rglob("*.cc"):
            if name in src.read_bytes():
                return True
    return False


def disabled_ops():
    """Families disabled by the transformer-only scope (CMake default)."""
    text = (ROOT / "cmake" / "NNTileKernelScope.cmake").read_text(
        errors="ignore")
    match = re.search(
        r'set\(NNTILE_DISABLED_KERNELS\s+"([^"]*)"', text)
    if not match:
        return set()
    return {item.strip() for item in match.group(1).split(";") if item.strip()}


def main():
    disabled = disabled_ops()
    ops = sorted(p.name for p in KERNELS.iterdir() if p.is_dir())
    chunked = chunked_ops()
    drivers = driver_ops()
    rows = []
    for op in ops:
        if op in disabled:
            kind = "disabled"
        elif op in chunked:
            kind = "chunked"
        elif op in drivers or op.replace("_inplace", "") in drivers:
            kind = "driver"
        elif not used_by_high_level(op):
            kind = "unused"
        else:
            kind = "classic"
        rows.append((op, kind))
    width = max(len(op) for op, _ in rows)
    counts = {"chunked": 0, "driver": 0, "classic": 0, "unused": 0,
              "disabled": 0}
    for op, kind in rows:
        counts[kind] += 1
        print(f"{op:<{width}}  {kind}")
    print()
    print(f"total {len(rows)}: chunked {counts['chunked']}, "
          f"driver {counts['driver']}, classic {counts['classic']}, "
          f"unused {counts['unused']}, disabled {counts['disabled']}")
    # NNHaul backend: `classic` rows are the review list; `driver` rows still need
    # the multi-chunk tile test to prove the buffers they see are chunks.
    return 0 if counts["classic"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
