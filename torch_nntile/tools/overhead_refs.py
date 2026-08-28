"""Shared reference numbers for cross-model overhead doc comparisons."""

from __future__ import annotations

from typing import Any

# GPT-2 10× reference (docs/dev/gpt2_hf_overhead_scale.md, A40 Aug 2026).
GPT2_REF: dict[str, Any] = {
    "ratios": {"xs": 0.99, "s": 0.96, "m": 0.94, "l": 0.94, "xl": 0.96},
    "long_steps": 100,
    "long_wall_s": 27.506,
    "long_wall_std_s": 0.018,
    "long_loss": 7.734033,
    "long_host_pct": 22,
}

# GPT-Neo 10× reference (docs/dev/gpt_neo_hf_overhead_scale.md, GPU 0).
GPT_NEO_REF: dict[str, Any] = {
    "ratios": {"xs": 0.99, "s": 0.97, "m": 0.95, "l": 0.95, "xl": 0.97},
    "long_wall_s": 27.561,
    "long_loss": 7.932405,
    "long_host_pct": 24,
}

# GPT-NeoX 10× reference (docs/dev/gpt_neox_hf_overhead_scale.md, GPU 0).
GPT_NEOX_REF: dict[str, Any] = {
    "ratios": {"xs": 1.14, "s": 1.04, "m": 1.03, "l": 1.00, "xl": 1.01},
    "long_wall_s": 29.019,
    "long_loss": 7.945045,
    "long_host_pct": 25,
}
