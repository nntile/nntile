#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/train_cuda_only.py
# CUDA-only train loop for CUDA vs nntile benches. Never import torch_nntile.

"""CUDA-only train helper for stock HF / CNN / DiT smokes.

PyTorch cannot use CUDA autograd in a process that imported
``torch_nntile`` (PrivateUse1). This script must stay CUDA-only: it
aborts if ``torch_nntile`` is already in ``sys.modules``.

Invoke with ``PYTHONPATH`` pointing at this ``examples/`` directory and
**without** ``libnntile`` / ``libtorch_nntile`` on ``LD_LIBRARY_PATH``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _examples_dir() -> Path:
    return Path(__file__).resolve().parent


def _abort_if_nntile(where: str) -> None:
    if "torch_nntile" in sys.modules:
        raise SystemExit(
            f"train_cuda_only: torch_nntile imported ({where}); abort"
        )


def disable_tf32() -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def train_loop(
    name: str,
    model,
    batch,
    loss_fn,
    steps: int,
    lr: float,
) -> None:
    import torch

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    last_loss = None
    t0 = time.perf_counter()
    for step in range(steps):
        t_iter0 = time.perf_counter()
        loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        step_loss = loss.detach()
        del loss
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        iter_s = time.perf_counter() - t_iter0
        print(
            f"timing torch iter {step + 1}/{steps} wall={iter_s:.3f}s",
            flush=True,
        )
        if step == steps - 1:
            last_loss = step_loss
        else:
            del step_loss
    wall_s = time.perf_counter() - t0
    if last_loss is None:
        raise RuntimeError(f"{name}: no steps ran")
    loss_val = float(last_loss.item())
    del last_loss
    print(f"[{name}] final loss={loss_val:.6f}", flush=True)
    print(f"[{name}] wall={wall_s:.3f}s  OK", flush=True)


def main() -> int:
    examples = _examples_dir()
    if str(examples) not in sys.path:
        sys.path.insert(0, str(examples))
    _abort_if_nntile("before torch")

    import torch

    _abort_if_nntile("after torch")

    from cnn_tiny_train_common import (
        classification_ce_loss,
        make_image_batch,
        make_segmentation_batch,
        segmentation_ce_loss,
    )
    from dit_hf_tiny_train_common import (
        diffusion_mse_loss,
        disable_dit_label_dropout,
        make_cifar_diffusion_batch,
    )
    from hf_tiny_train_common import (
        causal_ce_loss,
        load_hf_config_from_json,
        load_json_object,
        make_causal_batch,
        make_encoder_decoder_batch,
        make_mlm_batch,
        mlm_ce_loss,
        t5_ce_loss,
    )
    from train_lenet_tiny import TinyLeNet
    from train_mobilenet_tiny import TinyMobileNet
    from train_resnet_tiny import TinyResNet
    from train_unet_modern_tiny import TinyModernUNet
    from train_unet_tiny import TinyUNet
    from train_vgg_tiny import TinyVGG

    _abort_if_nntile("leaked via examples")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    args = parser.parse_args()
    disable_tf32()
    torch.manual_seed(args.seed)
    name = args.model
    steps = args.steps
    seed = args.seed

    if name in {
        "gpt-neo",
        "gpt-neox",
        "llama",
        "llama-gqa",
        "bert",
        "roberta",
        "t5",
    }:
        from transformers import (
            BertConfig,
            BertForMaskedLM,
            GPTNeoConfig,
            GPTNeoForCausalLM,
            GPTNeoXConfig,
            GPTNeoXForCausalLM,
            LlamaConfig,
            LlamaForCausalLM,
            RobertaConfig,
            RobertaForMaskedLM,
            T5Config,
            T5ForConditionalGeneration,
        )

        hf = {
            "gpt-neo": (
                examples / "gpt_neo_hf_tiny_config.json",
                GPTNeoConfig,
                GPTNeoForCausalLM,
                "eager",
            ),
            "gpt-neox": (
                examples / "gpt_neox_hf_tiny_config.json",
                GPTNeoXConfig,
                GPTNeoXForCausalLM,
                "sdpa",
            ),
            "llama": (
                examples / "llama_hf_tiny_config.json",
                LlamaConfig,
                LlamaForCausalLM,
                "sdpa",
            ),
            "llama-gqa": (
                examples / "llama_hf_tiny_gqa_config.json",
                LlamaConfig,
                LlamaForCausalLM,
                "sdpa",
            ),
            "bert": (
                examples / "bert_hf_tiny_config.json",
                BertConfig,
                BertForMaskedLM,
                "eager",
            ),
            "roberta": (
                examples / "roberta_hf_tiny_config.json",
                RobertaConfig,
                RobertaForMaskedLM,
                "eager",
            ),
            "t5": (
                examples / "t5_hf_tiny_config.json",
                T5Config,
                T5ForConditionalGeneration,
                "eager",
            ),
        }
        cfg_path, cfg_cls, model_cls, attn = hf[name]
        if args.config:
            cfg_path = Path(args.config)
        bs = args.batch_size if args.batch_size > 0 else 1
        sl = args.seq_len if args.seq_len > 0 else 16
        use_cache = None
        if name in {
            "gpt-neo",
            "gpt-neox",
            "llama",
            "llama-gqa",
            "t5",
        }:
            use_cache = False
        cfg = load_hf_config_from_json(
            cfg_path,
            cfg_cls,
            attn_implementation=attn,
            use_cache=use_cache,
        )
        model = model_cls(cfg).float().train()
        if name in {"gpt-neo", "gpt-neox", "llama", "llama-gqa"}:
            x, y = make_causal_batch(
                cfg.vocab_size,
                batch_size=bs,
                seq_len=sl,
                seed=seed,
            )
            batch = {"input_ids": x, "labels": y}
            loss_fn = causal_ce_loss
        elif name == "bert":
            x, y = make_mlm_batch(
                cfg.vocab_size,
                batch_size=bs,
                seq_len=sl,
                seed=seed,
            )
            batch = {
                "input_ids": x,
                "labels": y,
                "attention_mask": torch.ones_like(x),
                "token_type_ids": torch.zeros_like(x),
                "position_ids": (
                    torch.arange(sl, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(bs, -1)
                    .contiguous()
                ),
            }
            loss_fn = mlm_ce_loss
        elif name == "roberta":
            x, y = make_mlm_batch(
                cfg.vocab_size,
                batch_size=bs,
                seq_len=sl,
                seed=seed,
            )
            batch = {
                "input_ids": x,
                "labels": y,
                "attention_mask": torch.ones_like(x),
                "position_ids": (
                    torch.arange(sl, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(bs, -1)
                    .contiguous()
                ),
                "token_type_ids": torch.zeros_like(x),
            }
            loss_fn = mlm_ce_loss
        else:
            enc, dec, labels = make_encoder_decoder_batch(
                cfg.vocab_size,
                batch_size=bs,
                seq_len=sl,
                seed=seed,
            )
            batch = {
                "input_ids": enc,
                "decoder_input_ids": dec,
                "labels": labels,
            }
            loss_fn = t5_ce_loss
        lr = args.lr if args.lr > 0 else 1e-3
    elif name in {"lenet", "resnet", "vgg", "mobilenet"}:
        cfgs = {
            "lenet": (examples / "lenet_tiny_config.json", TinyLeNet),
            "resnet": (examples / "resnet_tiny_config.json", TinyResNet),
            "vgg": (examples / "vgg_tiny_config.json", TinyVGG),
            "mobilenet": (
                examples / "mobilenet_tiny_config.json",
                TinyMobileNet,
            ),
        }
        cfg_path, model_cls = cfgs[name]
        if args.config:
            cfg_path = Path(args.config)
        cfg = load_json_object(cfg_path)
        model = model_cls(cfg).float().train()
        cbs = args.batch_size if args.batch_size > 0 else 2
        batch = make_image_batch(
            batch_size=cbs,
            channels=int(cfg["in_channels"]),
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            num_classes=int(cfg["num_classes"]),
            seed=seed,
        )
        loss_fn = classification_ce_loss
        lr = args.lr if args.lr > 0 else 1e-2
    elif name in {"unet", "unet-modern"}:
        if name == "unet":
            cfg_path = examples / "unet_tiny_config.json"
        else:
            cfg_path = examples / "unet_modern_tiny_config.json"
        if args.config:
            cfg_path = Path(args.config)
        cfg = load_json_object(cfg_path)
        if name == "unet":
            model = TinyUNet(cfg).float().train()
        else:
            model = TinyModernUNet(cfg).float().train()
        cbs = args.batch_size if args.batch_size > 0 else 2
        batch = make_segmentation_batch(
            batch_size=cbs,
            channels=int(cfg["in_channels"]),
            height=int(cfg["height"]),
            width=int(cfg["width"]),
            num_classes=int(cfg["num_classes"]),
            seed=seed,
        )
        loss_fn = segmentation_ce_loss
        lr = args.lr if args.lr > 0 else 1e-2
    elif name == "dit":
        from diffusers import DiTTransformer2DModel

        cfg_path = examples / "dit_hf_tiny_config.json"
        if args.config:
            cfg_path = Path(args.config)
        fields = load_json_object(cfg_path)
        model = DiTTransformer2DModel(**fields).float().train()
        disable_dit_label_dropout(model)
        cfg = model.config
        sample_size = int(getattr(cfg, "sample_size", 16))
        in_channels = int(getattr(cfg, "in_channels", 3))
        num_timesteps = int(getattr(cfg, "num_embeds_ada_norm", 1000))
        num_classes = max(num_timesteps, 10)
        cbs = args.batch_size if args.batch_size > 0 else 2
        batch = make_cifar_diffusion_batch(
            batch_size=cbs,
            sample_size=sample_size,
            in_channels=in_channels,
            num_timesteps=num_timesteps,
            num_classes=num_classes,
            seed=seed,
        )
        loss_fn = diffusion_mse_loss
        lr = args.lr if args.lr > 0 else 1e-3
    else:
        raise SystemExit(f"unknown model {name}")

    _abort_if_nntile("during setup")

    batch = {k: v.to("cuda") for k, v in batch.items()}
    model = model.to("cuda")
    for param in model.parameters():
        param.requires_grad_(True)
    train_loop(name, model, batch, loss_fn, steps, lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
