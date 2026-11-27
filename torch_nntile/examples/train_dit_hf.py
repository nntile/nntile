#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_dit_hf.py
# Tiny stock Diffusers DiTTransformer2DModel smoke on cpu/nntile.

"""Tiny HF Diffusers DiT smoke (CIFAR-10 via ``datasets``).

Uses JSON config / checkpoint like other HF smokes::

    python torch_nntile/examples/train_dit_hf.py train \\
        --device nntile --seed 0 --config dit_hf_tiny_config.json \\
        --output-dir /tmp/dit_hf --steps 1

    python ... compare --checkpoint-a A.pt --checkpoint-b B.pt
"""

from __future__ import annotations

from pathlib import Path

from diffusers import DiTTransformer2DModel
from dit_hf_tiny_train_common import (
    diffusion_mse_loss, make_cifar_diffusion_batch, run_tiny_dit_hf_main)


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "dit_hf_tiny_config.json"


def main(argv: list[str] | None = None) -> int:
    def build_batch(cfg, args):
        sample_size = int(getattr(cfg, "sample_size", 16))
        in_channels = int(getattr(cfg, "in_channels", 3))
        num_timesteps = int(getattr(cfg, "num_embeds_ada_norm", 1000))
        # LabelEmbedding table is sized to num_embeds_ada_norm (+ CFG).
        num_classes = max(num_timesteps, 10)
        return make_cifar_diffusion_batch(
            batch_size=args.batch_size,
            sample_size=sample_size,
            in_channels=in_channels,
            num_timesteps=num_timesteps,
            num_classes=num_classes,
            seed=args.seed if args.seed is not None else 0,
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
        )

    return run_tiny_dit_hf_main(
        name="dit",
        argv=argv,
        default_config=_default_config(),
        model_cls=DiTTransformer2DModel,
        build_batch=build_batch,
        loss_fn=diffusion_mse_loss,
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
