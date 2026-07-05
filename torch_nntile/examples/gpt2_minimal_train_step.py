#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/gpt2_minimal_train_step.py
# Tiny GPT-2: forward, cross-entropy loss, backward, and SGD on device="nntile".

from __future__ import annotations

import torch
import torch_nntile
from transformers import GPT2Config, GPT2LMHeadModel

from torch_nntile.models.gpt2_hf_loader import load_hf_into_gpt2_lm_head
from torch_nntile.models.gpt2_minimal import GPT2LMHead
from torch_nntile.training import SGD, cross_entropy


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
    model.tie_weights()

    for param in model.parameters():
        param.requires_grad_(True)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()

    with torch.no_grad():
        ref_logits = hf(input_ids).logits
        ref_loss = torch.nn.functional.cross_entropy(
            ref_logits.view(-1, config.vocab_size),
            labels.view(-1),
        )

    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits, labels, reduction="mean")
    loss.backward()

    optimizer = SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    torch_nntile.wait()

    loss_cpu = loss.to("cpu")
    print(
        "forward match:",
        torch.allclose(logits.cpu(), ref_logits, rtol=1e-4, atol=1e-4),
    )
    print(
        "loss match:",
        torch.allclose(loss_cpu, ref_loss, rtol=1e-4, atol=1e-4),
    )
    wte_grad = model.transformer.wte.weight.grad
    print(
        "backward ok, wte grad norm:",
        wte_grad.norm().cpu().item() if wte_grad is not None else None,
    )


if __name__ == "__main__":
    main()
