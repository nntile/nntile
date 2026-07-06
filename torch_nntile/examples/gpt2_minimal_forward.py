#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/gpt2_minimal_forward.py
# Minimal GPT-2 forward and backward on device="nntile".

from __future__ import annotations

import torch
import torch_nntile
from transformers import GPT2Config, GPT2LMHeadModel

from torch_nntile.models.gpt2_hf_loader import load_hf_into_gpt2_lm_head
from torch_nntile.models.gpt2_minimal import GPT2LMHead


def main() -> None:
    torch_nntile.init_context(ncpu=1, ncuda=0, cpu_fallback=False)
    torch_nntile.restrict_cpu()

    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_positions=32,
        vocab_size=128,
        n_inner=256,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"

    torch.manual_seed(0)
    hf = GPT2LMHeadModel(config).eval().float()
    model = GPT2LMHead(config).eval().float()
    load_hf_into_gpt2_lm_head(model, hf)
    model = model.to("nntile")

    input_ids_cpu = torch.randint(0, config.vocab_size, (2, 8))
    input_ids = input_ids_cpu.to("nntile")
    with torch.no_grad():
        ref = hf(input_ids_cpu).logits
        out = model(input_ids).cpu()
    print("forward match:", torch.allclose(out, ref, rtol=1e-4, atol=1e-4))

    for p in model.parameters():
        p.requires_grad_(True)
    grad_out = torch.randn_like(out).to("nntile")
    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    logits.backward(grad_out)
    wte_grad = model.transformer.wte.weight.grad
    print(
        "backward ok, wte grad norm:",
        wte_grad.norm().cpu().item() if wte_grad is not None else None,
    )


if __name__ == "__main__":
    main()
