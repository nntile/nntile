#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_dit.py
# Tiny DiT smoke: host patchify, then nntile(nntile) noise-prediction MSE.

"""Short DiT training smoke: host patchify, then nntile(nntile) MSE.

Host prep (patchify NCHW, integer timesteps) happens before ``.to("nntile")``.
The nntile(nntile) model is ``torch_nntile.models.dit.DiT``.

Uses JSON config / checkpoint like ``train_llama.py``::

    python torch_nntile/examples/train_dit.py train \\
        --seed 0 --config dit_hf_tiny_config.json \\
        --output-dir /tmp/dit --steps 2
"""

from __future__ import annotations

from pathlib import Path

from dit_hf_tiny_train_common import make_synthetic_diffusion_batch
from nntile_tiny_train_common import run_tiny_nntile_main
from torch_nntile.models.dit import (
    DiT,
    DiTConfig,
    nchw_to_unpatchify_tokens,
    patchify_nchw,
)
from torch_nntile.nn.functional import add, mse_loss


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "dit_hf_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        patch = int(cfg.patch_size)
        raw = make_synthetic_diffusion_batch(
            batch_size=args.batch_size,
            sample_size=int(cfg.sample_size),
            in_channels=int(cfg.in_channels),
            num_timesteps=int(cfg.num_embeds_ada_norm),
            num_classes=int(cfg.num_embeds_ada_norm),
            seed=args.seed,
        )
        return {
            "patches": patchify_nchw(raw["noisy"], patch).contiguous(),
            "noise": nchw_to_unpatchify_tokens(
                raw["noise"], patch
            ).contiguous(),
            "timesteps": raw["timesteps"].contiguous(),
            "class_labels": raw["class_labels"].contiguous(),
        }

    def loss_fn(model, batch):
        pred = model(
            batch["patches"],
            batch["timesteps"],
            batch["class_labels"],
        )
        diff = add(pred, batch["noise"], alpha=1.0, beta=-1.0)
        return mse_loss(diff, scale=1.0 / float(pred.numel()))

    return run_tiny_nntile_main(
        name="dit",
        argv=argv,
        default_config=_default_config(),
        config_cls=DiTConfig,
        model_cls=DiT,
        build_batch=build_batch,
        loss_fn=loss_fn,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
