#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/gpt2_hf_out_of_box.py
# Stock HuggingFace GPT2LMHeadModel on device="nntile" (no custom model).

from __future__ import annotations

import torch
import torch_nntile  # noqa: F401 — registers device + HF compat
from transformers import GPT2Config, GPT2LMHeadModel

from torch_nntile.training import SGD, cross_entropy


def tiny_gpt2_config() -> GPT2Config:
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
    config._attn_implementation = "sdpa"
    return config


def main() -> None:
    torch_nntile.init_context(ncpu=1, ncuda=0, cpu_fallback=False)
    torch_nntile.restrict_cpu()

    config = tiny_gpt2_config()
    torch.manual_seed(0)
    ref = GPT2LMHeadModel(config).eval().float()
    model = GPT2LMHeadModel(config).eval().float()
    model.load_state_dict(ref.state_dict())
    model = model.to("nntile")

    for param in model.parameters():
        param.requires_grad_(True)
    for param in ref.parameters():
        param.requires_grad_(True)

    input_ids_cpu = torch.randint(0, config.vocab_size, (2, 8))
    labels_cpu = input_ids_cpu.clone()
    input_ids = input_ids_cpu.to("nntile")
    labels = labels_cpu.to("nntile")

    with torch.no_grad():
        ref_logits = ref(input_ids_cpu).logits
        ref_loss = torch.nn.functional.cross_entropy(
            ref_logits.view(-1, config.vocab_size),
            labels_cpu.view(-1),
        )

    model.zero_grad(set_to_none=True)
    logits = model(input_ids).logits
    loss = cross_entropy(logits, labels, reduction="mean")
    loss.backward()

    optimizer = SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    torch_nntile.wait()

    print(
        "forward match:",
        torch.allclose(logits.cpu(), ref_logits, rtol=1e-4, atol=1e-4),
    )
    print(
        "loss match:",
        torch.allclose(loss.to("cpu"), ref_loss, rtol=1e-4, atol=1e-4),
    )
    print(
        "wte grad norm:",
        model.transformer.wte.weight.grad.norm().cpu().item(),
    )


if __name__ == "__main__":
    main()
