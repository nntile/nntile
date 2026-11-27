#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# Compare aten dispatch registration vs CUDA parity policy.
"""Audit PrivateUse1 overrides against CUDA dispatch tables.

Example::

    python3 torch_nntile/tools/audit_cuda_parity_registration.py \\
        --ops native_layer_norm linear rms_norm matmul contiguous

Exit code 1 if any listed op gains an unexpected PrivateUse1 registration
after ``import torch_nntile`` (e.g. shadowing CompositeImplicitAutograd).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def dump_table(op: str, *, import_nntile: bool) -> str:
    script = f"""
import torch
{"import torch_nntile" if import_nntile else ""}
print(torch._C._dispatch_dump_table("{op}"))
"""
    out = subprocess.check_output(
        [sys.executable, "-c", script],
        text=True,
    )
    return out


def privateuse1_rows(table: str) -> list[str]:
    rows = []
    for line in table.splitlines():
        if "PrivateUse1" in line or "AutogradPrivateUse1" in line:
            rows.append(line.strip())
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=[
            "native_layer_norm",
            "linear",
            "matmul",
            "rms_norm",
            "contiguous",
            "chunk",
            "gelu",
        ],
        help="aten op names without the aten:: prefix",
    )
    args = parser.parse_args()

    # Ops that must never gain a PrivateUse1 *device* kernel
    # (CUDA uses composite). AutogradPrivateUse1 is allowed for
    # native_layer_norm (math via BN / affine so backward is subops).
    composite_only = {
        "linear",
        "matmul",
        "rms_norm",
        "layer_norm",
        "native_layer_norm",
        "chunk",
        "split",
        "narrow",
        "select.int",
    }
    allow_autograd_privateuse1 = {"contiguous", "native_layer_norm"}

    failed = False
    for op in args.ops:
        schema = f"aten::{op}"
        before = dump_table(schema, import_nntile=False)
        after = dump_table(schema, import_nntile=True)
        pu_before = privateuse1_rows(before)
        pu_after = privateuse1_rows(after)
        print(f"=== {schema} ===")
        if pu_after != pu_before:
            print("  PrivateUse1 changed after import torch_nntile:")
            for row in pu_after:
                if row not in pu_before:
                    print(f"    + {row}")
            for row in pu_before:
                if row not in pu_after:
                    print(f"    - {row}")
        else:
            print("  PrivateUse1 unchanged.")
        device_pu = [
            row
            for row in pu_after
            if "PrivateUse1" in row
            and "AutogradPrivateUse1" not in row
            and "[default backend kernel]" not in row
            and "[backend fallback]" not in row
        ]
        autograd_pu = [
            row
            for row in pu_after
            if "AutogradPrivateUse1" in row
            and "[autograd kernel]" not in row
        ]
        if op in composite_only and device_pu:
            print(
                f"  FAIL: {schema} must stay composite "
                f"(no PrivateUse1 device kernel); "
                f"see docs/dev/torch_nntile_cuda_parity_policy.md",
                file=sys.stderr,
            )
            failed = True
        if autograd_pu and op not in allow_autograd_privateuse1:
            if op in composite_only:
                print(
                    f"  FAIL: {schema} must not register "
                    f"AutogradPrivateUse1; "
                    f"see docs/dev/torch_nntile_cuda_parity_policy.md",
                    file=sys.stderr,
                )
                failed = True
        if re.search(r"CompositeImplicitAutograd", before) and pu_after:
            print(
                f"  WARN: CUDA uses CompositeImplicitAutograd but nntile "
                f"registers PrivateUse1 — likely a parity regression.",
                file=sys.stderr,
            )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
