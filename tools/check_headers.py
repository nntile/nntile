#!/usr/bin/env python3
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
#
# @file tools/check_headers.py
# Enforce the NNTile 2.0 legal header and @version on tracked files.
#
# @version 2.0.0

"""Check that every authored tracked file carries the NNTile 2.0 header.

Exceptions (documented in STYLE_GUIDE.md):
- external/ vendored dependencies and LICENSE files keep their own licenses;
- binary assets (images) cannot carry a text header;
- JSON data/config files cannot carry comments;
- third-party files with their own license (cmake/CodeCoverage.cmake) carry
  that license plus an SPDX line and the @version marker instead.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

VERSION = "2.0.0"
THIRD_PARTY = {"cmake/CodeCoverage.cmake"}
SKIP_PREFIX = ("external/", "images/")
SKIP_EXACT = {"LICENSE", "nnhaul/LICENSE"}
SKIP_SUFFIX = (".json", ".png", ".pdf", ".pyc", ".so", ".o")


def tracked() -> list[str]:
    return subprocess.check_output(["git", "ls-files"], text=True).split()


def main() -> int:
    missing: list[str] = []
    bad_version: list[str] = []
    for name in tracked():
        if (
            name.startswith(SKIP_PREFIX)
            or name in SKIP_EXACT
            or name.endswith(SKIP_SUFFIX)
        ):
            continue
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        head = text[:6000]
        if name in THIRD_PARTY:
            if "@version 2.0.0" not in head or "SPDX-License-Identifier" not in head:
                missing.append(name)
            continue
        if "@copyright" not in head:
            missing.append(name)
            continue
        match = re.search(r"@version\s+([0-9][0-9A-Za-z.+-]*)", text)
        if not match or match.group(1) != VERSION:
            bad_version.append(name)

    if missing or bad_version:
        for name in missing:
            print(f"missing 2.0 legal header: {name}", file=sys.stderr)
        for name in bad_version:
            print(f"header version is not {VERSION}: {name}", file=sys.stderr)
        return 1
    print("headers: every authored tracked file is 2.0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
