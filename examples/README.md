# NNTile C++ examples

Runnable programs under `examples/` show the graph API (forward, autograd,
incremental training phases) and small Llama / GPT-2 workflows.

Build all examples from the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Executables land in `build/examples/`. Python helpers in this directory need
`numpy`; HuggingFace-based scripts also need `torch`, `transformers`, and
`safetensors`.

Set `BUILD_DIR` if your build tree is not `build/` (used by the demo shell
scripts below).

## Quick start: graph training demos

These scripts prepare a **tiny** `uint16` `train.bin`, then call the C++
trainers with `--tiny` (256 vocab, 2 layers, CPU-friendly `Context`). Training
runs for several **epochs** on the **same** file so you should see `loss=`
decrease across steps.

**Llama**

```bash
cmake --build build --target llama_graph_training
./examples/run_llama_graph_training_demo.sh
```

**GPT-2**

```bash
cmake --build build --target gpt2_graph_training gpt2_generate
./examples/run_gpt2_graph_training_demo.sh
```

Each script writes data under `build/examples/demo_data/<model>/` and prints a
short loss summary (first vs last step). Tune without editing the script:

```bash
EPOCHS=6 MAX_BATCHES=48 LR=0.005 ./examples/run_llama_graph_training_demo.sh
```

| Variable       | Default | Meaning                                      |
|----------------|---------|----------------------------------------------|
| `BUILD_DIR`    | `build` | CMake build directory                        |
| `DATA_DIR`     | auto    | Output folder for `train.bin` and logs       |
| `SEQ_LEN`      | `8`     | Passed to `--seq`                            |
| `BATCH_SIZE`   | `2`     | Passed to `--batch`                            |
| `NUM_BATCHES`  | `8`     | Batches stored in `train.bin`                |
| `EPOCHS`       | `4`     | Full passes over `train.bin`                 |
| `MAX_BATCHES`  | `32`    | Cap on optimizer steps (`0` = no cap)        |
| `LR`           | `0.003` | Learning rate                                |

Data prep only (offline, no HuggingFace):

```bash
python3 examples/prepare_tiny_train_bin.py \
    --output build/examples/demo_data/llama/train.bin \
    --seq-len 8 --batch-size 2 --num-batches 8
```

For **real text**, tokenize with
`wrappers/python/examples/causal_lm_data_preparation.py` and point
`--train-bin` at the generated `train.bin` (see comments in
`llama_graph_training.cc` / `gpt2_graph_training.cc`).

## Llama

| Artifact | Role |
|----------|------|
| `llama_graph_training` | Causal LM training on mmap `train.bin` (graph API, AdamW, incremental phases) |
| `run_llama_graph_training_demo.sh` | Tiny data + multi-epoch demo (above) |
| `llama_graph_training.cc` | Full CLI (`--tiny`, `--config`, `--load-weights`, …) |
| `llama_generate` | Autoregressive generation (KV cache) |
| `llama_generate.py` | HF checkpoint → NNTile weights + prompt tokenization |
| `llama_inference_server.py` | Minimal inference server (Python API) |

Example (manual training after demo data exists):

```bash
./build/examples/llama_graph_training \
    --train-bin build/examples/demo_data/llama/train.bin \
    --tiny --seq 8 --batch 2 --epochs 3 --max-batches 24 --lr 0.003
```

Example (generation):

```bash
python3 examples/llama_generate.py --model meta-llama/Llama-2-7b-hf \
    --output-dir /tmp/nntile_llama --prompt "Hello"
./build/examples/llama_generate \
    --config /tmp/nntile_llama/config.json \
    --weights /tmp/nntile_llama/weights.safetensors \
    --prompt-ids "$(cat /tmp/nntile_llama/prompt_ids.txt)"
```

## GPT-2

| Artifact | Role |
|----------|------|
| `gpt2_graph_training` | Causal LM training on mmap `train.bin` (same pattern as Llama) |
| `run_gpt2_graph_training_demo.sh` | Tiny data + multi-epoch demo |
| `gpt2_graph_training.cc` | Full CLI |
| `gpt2_generate` | Greedy generation (no KV cache; rebuilds graph each step) |
| `gpt2_generate.py` | HF GPT-2 → weights + prompt |
| `gpt2_inference_server.py` | Minimal inference server (Python API) |

Example (manual training):

```bash
./build/examples/gpt2_graph_training \
    --train-bin build/examples/demo_data/gpt2/train.bin \
    --tiny --seq 8 --batch 2 --epochs 3 --max-batches 24 --lr 0.003
```

Example (generation):

```bash
python3 examples/gpt2_generate.py --model gpt2 --output-dir /tmp/nntile_gpt2 \
    --prompt "The capital of France is"
./build/examples/gpt2_generate \
    --config /tmp/nntile_gpt2/config.json \
    --weights /tmp/nntile_gpt2/weights.safetensors \
    --prompt-ids "$(cat /tmp/nntile_gpt2/prompt_ids.txt)" \
    --max-tokens 32
```

Or let the binary invoke the Python converter:

```bash
./build/examples/gpt2_generate --model gpt2 --prompt "Hello" --max-tokens 16
```

## Other graph / autograd examples

| Binary | Description |
|--------|-------------|
| `autograd_add_example` | Minimal autograd on `NNGraph` |
| `graph_mlp_example` | MLP forward on graph API |
| `autograd_mlp_tile_compare_example` | MLP autograd vs tile reference |
| `linear_layer_example` | Linear layer forward |
| `deep_relu_forward` | Deep ReLU forward |
| `deep_relu_training` | ReLU MLP training with SGD / Adam / AdamW |

## Notes

- Training examples use `CONTEXT_NUM_CUDA = 0` so they run on CPU-only machines.
- `llama_generate` may use CUDA workers when available; see `llama_generate.cc`.
- Incremental training reuses tile state when shapes match; each step calls
  `finish_phase()`, `lower_and_compile()`, and `runtime.execute()` (see source
  comments in the training `.cc` files).
