#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_nntile_native_overhead.py
# Overhead train for torch_nntile.models (HF init, classic kernels).

"""Classic-kernel overhead trainer. HF is used only to initialize weights."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from hf_tiny_train_common import (
    configure_single_thread_host,
    load_hf_config_from_json,
    make_encoder_decoder_batch,
    make_mlm_batch,
)
from nntile_native_overhead_common import BatchDict, run_native_overhead
from torch_nntile.training import cross_entropy

IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_tokens(
    vocab_size: int,
    *,
    num_tokens: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.long, generator=generator
    )


def build_sequences(
    token_ids: torch.Tensor,
    *,
    example_len: int,
    max_sequences: int,
) -> torch.Tensor:
    n_seq = int(token_ids.numel()) // example_len
    n_seq = min(n_seq, max_sequences)
    usable = n_seq * example_len
    return token_ids[:usable].view(n_seq, example_len)


def make_batches(
    sequences: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[torch.Tensor]:
    n = sequences.shape[0]
    order = torch.arange(n)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        order = order[torch.randperm(n, generator=generator)]
    batches: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        if idx.numel() == 0:
            continue
        batches.append(sequences[idx].contiguous())
    return batches


def causal_epochs(
    vocab_size: int,
    args: argparse.Namespace,
) -> list[list[BatchDict]]:
    example_len = args.seq_len + 1
    num_tokens = example_len * args.max_sequences
    sequences = build_sequences(
        make_synthetic_tokens(
            vocab_size, num_tokens=num_tokens, seed=args.data_seed
        ),
        example_len=example_len,
        max_sequences=args.max_sequences,
    )
    epochs: list[list[BatchDict]] = []
    for epoch in range(args.epochs):
        packed = make_batches(
            sequences,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
            shuffle=not args.no_shuffle,
        )
        epoch_data: list[BatchDict] = []
        for batch in packed:
            inputs, labels = batch[:, :-1].clone(), batch[:, 1:].clone()
            epoch_data.append({"input_ids": inputs, "labels": labels})
        epochs.append(epoch_data)
    return epochs


def mlm_epochs(
    vocab_size: int,
    args: argparse.Namespace,
) -> list[list[BatchDict]]:
    epochs: list[list[BatchDict]] = []
    for epoch in range(args.epochs):
        steps: list[BatchDict] = []
        for step in range(args.max_sequences):
            input_ids, labels = make_mlm_batch(
                vocab_size,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=args.data_seed + epoch * args.max_sequences + step,
            )
            steps.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "token_type_ids": torch.zeros_like(input_ids),
                    "position_ids": (
                        torch.arange(args.seq_len, dtype=torch.long)
                        .unsqueeze(0)
                        .expand(args.batch_size, -1)
                        .contiguous()
                    ),
                }
            )
        epochs.append(steps)
    return epochs


def t5_epochs(
    vocab_size: int,
    args: argparse.Namespace,
) -> list[list[BatchDict]]:
    epochs: list[list[BatchDict]] = []
    for epoch in range(args.epochs):
        steps: list[BatchDict] = []
        for step in range(args.max_sequences):
            enc, dec, labels = make_encoder_decoder_batch(
                vocab_size,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                seed=args.data_seed + epoch * args.max_sequences + step,
            )
            steps.append(
                {
                    "input_ids": enc,
                    "decoder_input_ids": dec,
                    "labels": labels,
                }
            )
        epochs.append(steps)
    return epochs


def causal_loss(model: torch.nn.Module, batch: BatchDict) -> torch.Tensor:
    logits = model(batch["input_ids"])
    return cross_entropy(logits, batch["labels"], reduction="mean")


def mlm_loss(model: torch.nn.Module, batch: BatchDict) -> torch.Tensor:
    kwargs: dict = {"input_ids": batch["input_ids"]}
    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"]
    if "position_ids" in batch:
        kwargs["position_ids"] = batch["position_ids"]
    logits = model(**kwargs)
    return cross_entropy(
        logits,
        batch["labels"],
        reduction="mean",
        ignore_index=IGNORE_INDEX,
    )


def t5_loss(model: torch.nn.Module, batch: BatchDict) -> torch.Tensor:
    logits = model(batch["input_ids"], batch["decoder_input_ids"])
    return cross_entropy(logits, batch["labels"], reduction="mean")


def llama_from_hf(hf: torch.nn.Module) -> torch.nn.Module:
    from torch_nntile.models.llama import LlamaCausal
    from torch_nntile.models.llama_hf_loader import (
        llama_config_from_hf,
        load_hf_into_llama_causal,
    )

    model = LlamaCausal(llama_config_from_hf(hf.config)).float()
    load_hf_into_llama_causal(model, hf)
    model.train()
    return model


def gpt_neo_from_hf(hf: torch.nn.Module) -> torch.nn.Module:
    from torch_nntile.models.gpt_neo import GPTNeoCausal
    from torch_nntile.models.gpt_neo_hf_loader import (
        gpt_neo_config_from_hf,
        load_hf_into_gpt_neo_causal,
    )

    model = GPTNeoCausal(gpt_neo_config_from_hf(hf.config)).float()
    load_hf_into_gpt_neo_causal(model, hf)
    model.train()
    return model


def gpt_neox_from_hf(hf: torch.nn.Module) -> torch.nn.Module:
    from torch_nntile.models.gpt_neox import GPTNeoXCausal
    from torch_nntile.models.gpt_neox_hf_loader import (
        gpt_neox_config_from_hf,
        load_hf_into_gpt_neox_causal,
    )

    model = GPTNeoXCausal(gpt_neox_config_from_hf(hf.config)).float()
    load_hf_into_gpt_neox_causal(model, hf)
    model.train()
    return model


def bert_from_hf(hf: torch.nn.Module) -> torch.nn.Module:
    from torch_nntile.models.bert import BertMlm
    from torch_nntile.models.bert_hf_loader import (
        bert_config_from_hf,
        load_hf_into_bert_mlm,
    )

    model = BertMlm(bert_config_from_hf(hf.config)).float()
    load_hf_into_bert_mlm(model, hf)
    model.train()
    return model


def roberta_from_hf(hf: torch.nn.Module) -> torch.nn.Module:
    from torch_nntile.models.roberta import RobertaMlm
    from torch_nntile.models.roberta_hf_loader import (
        load_hf_into_roberta_mlm,
        roberta_config_from_hf,
    )

    model = RobertaMlm(roberta_config_from_hf(hf.config)).float()
    load_hf_into_roberta_mlm(model, hf)
    model.train()
    return model


def t5_from_hf(hf: torch.nn.Module) -> torch.nn.Module:
    from torch_nntile.models.t5 import T5ForConditionalGeneration
    from torch_nntile.models.t5_hf_loader import (
        load_hf_into_t5,
        t5_config_from_hf,
    )

    model = T5ForConditionalGeneration(t5_config_from_hf(hf.config)).float()
    load_hf_into_t5(model, hf)
    model.train()
    return model


def _hf_pair(name: str) -> tuple[type, type, str | None]:
    if name == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        return LlamaConfig, LlamaForCausalLM, "sdpa"
    if name == "gpt_neo":
        from transformers import GPTNeoConfig, GPTNeoForCausalLM

        return GPTNeoConfig, GPTNeoForCausalLM, "eager"
    if name == "gpt_neox":
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        return GPTNeoXConfig, GPTNeoXForCausalLM, "sdpa"
    if name == "bert":
        from transformers import BertConfig, BertForMaskedLM

        return BertConfig, BertForMaskedLM, "eager"
    if name == "roberta":
        from transformers import RobertaConfig, RobertaForMaskedLM

        return RobertaConfig, RobertaForMaskedLM, "eager"
    if name == "t5":
        from transformers import T5Config, T5ForConditionalGeneration

        return T5Config, T5ForConditionalGeneration, "eager"
    raise SystemExit(f"unknown family: {name}")


NATIVE: dict[str, Callable[[torch.nn.Module], torch.nn.Module]] = {
    "llama": llama_from_hf,
    "gpt_neo": gpt_neo_from_hf,
    "gpt_neox": gpt_neox_from_hf,
    "bert": bert_from_hf,
    "roberta": roberta_from_hf,
    "t5": t5_from_hf,
}

KIND = {
    "llama": "causal",
    "gpt_neo": "causal",
    "gpt_neox": "causal",
    "bert": "mlm",
    "roberta": "mlm",
    "t5": "t5",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train"])
    parser.add_argument(
        "--family",
        required=True,
        choices=list(NATIVE),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ncpu", type=int, default=0)
    parser.add_argument("--ncuda", type=int, default=1)
    parser.add_argument("--restrict-cuda", action="store_true")
    parser.add_argument("--restrict-cpu", action="store_true")
    parser.add_argument("--wait-after-run", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_single_thread_host()
    args.data_seed = (
        int(args.data_seed) if args.data_seed is not None else args.seed
    )
    hf_cfg_cls, hf_model_cls, attn = _hf_pair(args.family)
    hf_config = load_hf_config_from_json(
        Path(args.config),
        hf_cfg_cls,
        attn_implementation=attn,
        use_cache=False,
    )
    set_seed(args.seed)
    hf = hf_model_cls(hf_config).float()
    hf.train()
    cpu_model = NATIVE[args.family](hf)
    del hf
    vocab = int(hf_config.vocab_size)
    kind = KIND[args.family]
    if kind == "causal":
        epochs = causal_epochs(vocab, args)
        loss_fn = causal_loss
    elif kind == "mlm":
        epochs = mlm_epochs(vocab, args)
        loss_fn = mlm_loss
    else:
        epochs = t5_epochs(vocab, args)
        loss_fn = t5_loss
    return run_native_overhead(
        name=args.family,
        args=args,
        cpu_model=cpu_model,
        epoch_batches_cpu=epochs,
        loss_fn=loss_fn,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    raise SystemExit(main())
