#!/usr/bin/env python3
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/bench_cuda_vs_nntile_2gb.py
# CUDA vs nntile on >=2 GiB FP32 configs (separate processes).

"""Compare CUDA vs device=nntile on 2 GiB stock-model recipes.

Torch cannot use CUDA autograd and PrivateUse1 ``nntile`` in one
process. This orchestrator launches **two Python processes per model**
(CUDA first, then nntile) and prints a markdown table of final loss,
train-loop wall, and nvidia-smi VRAM delta.

CUDA children must never import ``torch_nntile``. They run
``train_cuda_only.py`` (or ``train_gpt2_hf.py --device cuda``) with
``PYTHONPATH`` = this ``examples/`` directory and without libnntile on
``LD_LIBRARY_PATH``. Nntile children use the existing ``train_*``
scripts with ``--device nntile --ncpu 0 --ncuda 1 --restrict-cuda``.

See ``docs/dev/cuda_vs_nntile_2gb.md``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CFG_DIR = HERE / "2gb"
CUDA_TRAIN = HERE / "train_cuda_only.py"


def _gpu_id(args_gpu: str) -> str:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if args_gpu:
        return args_gpu.split(",")[0]
    if env:
        return env.split(",")[0]
    raise SystemExit(
        "Pick an idle GPU: pass --gpu N or set CUDA_VISIBLE_DEVICES"
    )


def gpu_mem_mib(gpu: str) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(out.strip())


def _without_nntile_libs(ld: str) -> str:
    parts: list[str] = []
    for part in ld.split(":"):
        if not part:
            continue
        if "nntile/build" in part:
            continue
        stripped = part.rstrip("/")
        if stripped.endswith("/nntile"):
            continue
        if stripped.endswith("/torch_nntile"):
            continue
        parts.append(part)
    return ":".join(parts)


def cuda_env(*, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["LD_LIBRARY_PATH"] = _without_nntile_libs(
        env.get("LD_LIBRARY_PATH", "")
    )
    return env


def nntile_env(*, gpu: str, repo: Path, build: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "torch_nntile")
    starpu = os.environ.get("STARPU_LIB", "/opt/starpu/lib")
    extra = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{build / 'nntile'}:{build / 'torch_nntile'}:{starpu}"
        + (f":{extra}" if extra else "")
    )
    env["STARPU_SILENT"] = "1"
    env["STARPU_FXT_TRACE"] = "0"
    env["STARPU_WORKERS_NOBIND"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    return env


def cuda_train(
    model: str,
    config: Path,
    *,
    steps: int,
    seed: int,
    batch: int,
    seq_len: int,
    hf: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(CUDA_TRAIN),
        "--model",
        model,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--config",
        str(config),
        "--batch-size",
        str(batch),
    ]
    if hf:
        cmd += ["--seq-len", str(seq_len)]
    return cmd


def hf_nntile(
    script: str,
    config: Path,
    *,
    steps: int,
    seed: int,
    batch: int,
    seq_len: int,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / script),
        "train",
        "--device",
        "nntile",
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch),
        "--config",
        str(config),
        "--ncpu",
        "0",
        "--ncuda",
        "1",
        "--restrict-cuda",
    ]


def cnn_nntile(
    script: str,
    config: Path,
    *,
    steps: int,
    seed: int,
    batch: int,
) -> list[str]:
    return [
        sys.executable,
        str(HERE / script),
        "train",
        "--device",
        "nntile",
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch),
        "--config",
        str(config),
        "--ncpu",
        "0",
        "--ncuda",
        "1",
        "--restrict-cuda",
    ]


def gpt2_cmd(
    device: str,
    *,
    steps: int,
    seed: int,
    batch: int,
    seq_len: int,
    output_root: Path,
) -> list[str]:
    out = output_root / f"gpt2_{device}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HERE / "train_gpt2_hf.py"),
        "train",
        "--device",
        device,
        "--seed",
        str(seed),
        "--data-seed",
        str(seed),
        "--epochs",
        "1",
        "--max-sequences",
        str(steps * batch),
        "--batch-size",
        str(batch),
        "--seq-len",
        str(seq_len),
        "--config",
        str(CFG_DIR / "gpt2.json"),
        "--output-dir",
        str(out),
        "--no-shuffle",
        "--ncpu",
        "0",
        "--ncuda",
        "1",
    ]
    if device == "cuda":
        cmd.append("--disable-tf32")
    else:
        cmd.append("--restrict-cuda")
    return cmd


def build_jobs(
    *,
    steps: int,
    seed: int,
    hf_bs: int,
    hf_seq: int,
    cnn_bs: int,
    output_root: Path,
    hf_only: bool = False,
) -> list[tuple[str, list[str], list[str]]]:
    cfg = CFG_DIR
    hf = dict(steps=steps, seed=seed, batch=hf_bs, seq_len=hf_seq)
    cnn = dict(steps=steps, seed=seed, batch=cnn_bs)
    hf_jobs: list[tuple[str, list[str], list[str]]] = [
        (
            "GPT-2 HF",
            gpt2_cmd(
                "cuda",
                steps=steps,
                seed=seed,
                batch=hf_bs,
                seq_len=hf_seq,
                output_root=output_root,
            ),
            gpt2_cmd(
                "nntile",
                steps=steps,
                seed=seed,
                batch=hf_bs,
                seq_len=hf_seq,
                output_root=output_root,
            ),
        ),
        (
            "GPT-Neo HF",
            cuda_train(
                "gpt-neo", cfg / "gpt_neo.json", hf=True, **hf
            ),
            hf_nntile("train_gpt_neo_hf.py", cfg / "gpt_neo.json", **hf),
        ),
        (
            "GPT-NeoX HF",
            cuda_train(
                "gpt-neox", cfg / "gpt_neox.json", hf=True, **hf
            ),
            hf_nntile("train_gpt_neox_hf.py", cfg / "gpt_neox.json", **hf),
        ),
        (
            "Llama HF",
            cuda_train("llama", cfg / "llama.json", hf=True, **hf),
            hf_nntile("train_llama_hf.py", cfg / "llama.json", **hf),
        ),
        (
            "Llama HF GQA",
            cuda_train(
                "llama-gqa", cfg / "llama_gqa.json", hf=True, **hf
            ),
            hf_nntile("train_llama_hf.py", cfg / "llama_gqa.json", **hf),
        ),
        (
            "BERT HF",
            cuda_train("bert", cfg / "bert.json", hf=True, **hf),
            hf_nntile("train_bert_hf.py", cfg / "bert.json", **hf),
        ),
        (
            "RoBERTa HF",
            cuda_train("roberta", cfg / "roberta.json", hf=True, **hf),
            hf_nntile("train_roberta_hf.py", cfg / "roberta.json", **hf),
        ),
        (
            "T5 HF",
            cuda_train("t5", cfg / "t5.json", hf=True, **hf),
            hf_nntile("train_t5_hf.py", cfg / "t5.json", **hf),
        ),
    ]
    if hf_only:
        return hf_jobs
    cnn_jobs: list[tuple[str, list[str], list[str]]] = [
        (
            "LeNet",
            cuda_train(
                "lenet", cfg / "lenet.json", hf=False, **cnn, seq_len=0
            ),
            cnn_nntile("train_lenet_tiny.py", cfg / "lenet.json", **cnn),
        ),
        (
            "ResNet",
            cuda_train(
                "resnet", cfg / "resnet.json", hf=False, **cnn, seq_len=0
            ),
            cnn_nntile("train_resnet_tiny.py", cfg / "resnet.json", **cnn),
        ),
        (
            "VGG",
            cuda_train("vgg", cfg / "vgg.json", hf=False, **cnn, seq_len=0),
            cnn_nntile("train_vgg_tiny.py", cfg / "vgg.json", **cnn),
        ),
        (
            "MobileNet",
            cuda_train(
                "mobilenet",
                cfg / "mobilenet.json",
                hf=False,
                **cnn,
                seq_len=0,
            ),
            cnn_nntile(
                "train_mobilenet_tiny.py", cfg / "mobilenet.json", **cnn
            ),
        ),
        (
            "UNet",
            cuda_train("unet", cfg / "unet.json", hf=False, **cnn, seq_len=0),
            cnn_nntile("train_unet_tiny.py", cfg / "unet.json", **cnn),
        ),
        (
            "UNet modern",
            cuda_train(
                "unet-modern",
                cfg / "unet_modern.json",
                hf=False,
                **cnn,
                seq_len=0,
            ),
            cnn_nntile(
                "train_unet_modern_tiny.py",
                cfg / "unet_modern.json",
                **cnn,
            ),
        ),
        (
            "DiT HF",
            cuda_train("dit", cfg / "dit.json", hf=False, **cnn, seq_len=0),
            cnn_nntile("train_dit_hf.py", cfg / "dit.json", **cnn),
        ),
    ]
    return hf_jobs + cnn_jobs


def _seconds(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match is not None else "—"


def parse_metrics(text: str) -> tuple[str, str, str, str, str, str, str]:
    losses = re.findall(r"loss=([0-9.+-eE]+)", text)
    loss = losses[-1] if losses else "FAIL"
    match = re.search(r"\[.+\] wall=([0-9.]+)s", text)
    if match is None:
        match = re.search(
            r"timing (?:torch|nntile) train wall[^:]*: ([0-9.]+)s",
            text,
        )
    wall = match.group(1) if match is not None else "FAIL"
    rec_nntile = _seconds(
        text, r"timing nntile record\(nntile\): ([0-9.]+)s"
    )
    rec_torch = _seconds(
        text, r"timing nntile record\(torch\): ([0-9.]+)s"
    )
    compile_s = _seconds(text, r"timing nntile compile[^:]*: ([0-9.]+)s")
    run_s = _seconds(text, r"timing nntile run: ([0-9.]+)s")
    wait_s = _seconds(text, r"timing nntile wait[^:]*: ([0-9.]+)s")
    return loss, wall, rec_nntile, rec_torch, compile_s, run_s, wait_s


def run_one(
    cmd: list[str],
    env: dict[str, str],
    *,
    gpu: str,
) -> tuple[int, str, int]:
    baseline = gpu_mem_mib(gpu)
    samples: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                samples.append(gpu_mem_mib(gpu))
            except Exception:
                pass
            stop.wait(0.15)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
        env=env,
    )
    stop.set()
    thread.join(timeout=2)
    peak = max(samples) if samples else baseline
    vram = max(0, peak - baseline)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out[-8000:], flush=True)
    return proc.returncode, out, vram


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CUDA vs nntile 2 GiB comparison (separate processes)",
    )
    parser.add_argument(
        "--gpu",
        default="",
        help="Physical GPU index (or set CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--cnn-batch-size", type=int, default=4)
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="Run only HuggingFace transformer jobs (skip CNN / DiT)",
    )
    parser.add_argument(
        "--build-dir",
        default="",
        help="NNTile build dir (default: $NNTILE_BUILD_DIR or <repo>/build)",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="GPT-2 checkpoint dir (default: a temp directory)",
    )
    args = parser.parse_args()
    gpu = _gpu_id(args.gpu)
    if args.build_dir:
        build = Path(args.build_dir).resolve()
    elif os.environ.get("NNTILE_BUILD_DIR"):
        build = Path(os.environ["NNTILE_BUILD_DIR"]).resolve()
    else:
        build = REPO / "build"
    if args.output_root:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = Path(tempfile.mkdtemp(prefix="cuda_vs_nntile_2gb_"))

    jobs = build_jobs(
        steps=args.steps,
        seed=args.seed,
        hf_bs=args.hf_batch_size,
        hf_seq=args.seq_len,
        cnn_bs=args.cnn_batch_size,
        output_root=output_root,
        hf_only=args.hf_only,
    )
    env_c = cuda_env(gpu=gpu)
    env_n = nntile_env(gpu=gpu, repo=REPO, build=build)
    rows: list[
        tuple[
            str, str, str, str, str, str, str, str, str, str, str, str
        ]
    ] = []
    failed = 0
    print(f"# GPU={gpu}  build={build}  output={output_root}", flush=True)
    for name, cuda_cmd, nntile_cmd in jobs:
        print(f"\n==== {name} cuda ====", flush=True)
        rc_c, out_c, vram_c = run_one(cuda_cmd, env_c, gpu=gpu)
        loss_c, wall_c, _, _, _, _, _ = parse_metrics(out_c)
        if rc_c != 0:
            failed += 1
            if loss_c == "FAIL":
                loss_c = f"FAIL({rc_c})"
        print(
            f"  cuda  loss={loss_c}  wall={wall_c}s  vram={vram_c}MiB",
            flush=True,
        )

        print(f"==== {name} nntile ====", flush=True)
        rc_n, out_n, vram_n = run_one(nntile_cmd, env_n, gpu=gpu)
        loss_n, wall_n, rec_nn, rec_th, cmp_n, run_n, wait_n = (
            parse_metrics(out_n)
        )
        if rc_n != 0:
            failed += 1
            if loss_n == "FAIL":
                loss_n = f"FAIL({rc_n})"
        print(
            f"  nntile loss={loss_n}  record(nntile)={rec_nn}s "
            f"record(torch)={rec_th}s  compile={cmp_n}s "
            f"run={run_n}s  wait={wait_n}s  wall={wall_n}s  "
            f"vram={vram_n}MiB",
            flush=True,
        )
        rows.append(
            (
                name,
                loss_c,
                loss_n,
                str(vram_c),
                str(vram_n),
                wall_c,
                rec_nn,
                rec_th,
                cmp_n,
                run_n,
                wait_n,
                wall_n,
            )
        )

    print(
        "\n# CUDA vs nntile, 2 GiB configs, "
        f"{args.steps} steps, HF batch {args.hf_batch_size} "
        f"seq {args.seq_len}, CNN batch {args.cnn_batch_size}, "
        "1 GPU\n"
    )
    print(
        "| Model | CUDA loss | nntile loss | CUDA VRAM | nntile VRAM "
        "| CUDA wall | record(nntile) | record(torch) | nntile compile "
        "| nntile run | nntile wait | nntile wall |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for name, lc, ln, vc, vn, wc, rec_nn, rec_th, cmp, run, wait, wn in (
        rows
    ):
        print(
            f"| {name} | {lc} | {ln} | {vc} MiB | {vn} MiB "
            f"| {wc} s | {rec_nn} s | {rec_th} s | {cmp} s | {run} s "
            f"| {wait} s | {wn} s |"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
