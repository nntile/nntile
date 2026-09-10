#!/usr/bin/env bash
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file .github/scripts/check-model-classic-nn.sh
#
# Production torch_nntile.models must not call stock torch.nn compute
# (Linear / LayerNorm / Embedding / F.linear / F.relu / ...). Those lower
# to torch-native TORCH_* ops. Use torch_nntile.nn instead.
set -euo pipefail

root="${1:-.}"
cd "$root"

models_dir="torch_nntile/torch_nntile/models"
if [ ! -d "$models_dir" ]; then
    echo "skip: $models_dir not found"
    exit 0
fi

bad=/tmp/bad-model-classic-nn.txt
rm -f "$bad"

# HF loaders and host layout helpers may mention torch.nn.Linear types.
# CPU Mixer reference keeps stock nn as the parity oracle.
git grep -nE \
    'nn\.(Linear|LayerNorm|Embedding|ReLU|GELU|SiLU)|F\.(linear|relu|gelu|silu|scaled_dot_product_attention)|torch\.nn\.functional\.(relu|gelu|silu|linear)|torch\.ops\.aten\.' \
    -- "$models_dir" \
    ':!*hf_loader.py' \
    ':!*hf_rope_layout.py' \
    >>"$bad" 2>/dev/null || true

if [ -s "$bad" ]; then
    # Allow the CPU Mixer reference classes only.
    filtered=/tmp/bad-model-classic-nn-filtered.txt
    grep -vE 'mlp_mixer.py:.*(MlpMixerCpu|_CpuMixer|nn\.Linear|nn\.GELU|nn\.LayerNorm)' \
        "$bad" >"$filtered" || true
    # The CPU block still matches nn.Linear. Strip lines after class MlpMixerCpu.
    python3 - <<'PY' "$bad" "$filtered"
import re
import sys
src, dst = sys.argv[1], sys.argv[2]
keep = []
cpu_block = False
for line in open(src, encoding="utf-8"):
    path = line.split(":", 1)[0]
    if path.endswith("mlp_mixer.py"):
        # Skip CPU-reference section: from class MlpMixerCpu to EOF.
        m = re.search(r":(\d+):", line)
        if m and int(m.group(1)) >= 131:
            continue
    keep.append(line)
open(dst, "w", encoding="utf-8").writelines(keep)
PY
    if [ -s "$filtered" ]; then
        echo "::error::torch_nntile.models must use torch_nntile.nn, not stock torch.nn compute:"
        cat "$filtered"
        exit 1
    fi
fi

echo "Python models do not use stock torch.nn compute"
