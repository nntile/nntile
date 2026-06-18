# NNTile C++ examples

Runnable programs under `examples/` show the graph API (forward, autograd,
incremental training phases) and small Llama / GPT-2 workflows.


## Shared C++ utilities

- `json_config_helpers.hh` — `config_get_int` / `config_get_float` for JSON configs
  (used by GPT-2 and Llama graph examples).
- `gpt2_config_json.hh` — `load_gpt2_config_json` / `save_gpt2_config_json` (HF + NNTile keys).
- `tiling_config_json.hh` — `load_tiling_json` / `save_tiling_json` (`default` + `layers` in `tiling.json`).
- `gpt2_axis_naming.hh` — name axis groups for GPT-2 graph training before applying tiling.
- PyTorch bridge: `torch_nntile.set_axis_group_name` / `set_axis_group_tiling` — same axis-group model from `device="nntile"` (see [docs/torch_nntile.md](../../docs/torch_nntile.md)).
- `t5_config_json.hh` — `load_t5_config_json` / `save_t5_config_json` for T5 graph examples.
- `gptneo_config_json.hh` — load/save for examples; HF `attention_types` parsing lives in `include/nntile/model/gptneo/gptneo_config_json.hh`.

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

**T5**

```bash
cmake --build build --target t5_graph_training t5_generate
./examples/run_t5_graph_training_demo.sh
```

**BERT / RoBERTa** (in-process toy MLM batches; no `train.bin`)

```bash
cmake --build build --target bert_graph_training roberta_graph_training
./examples/run_bert_graph_training_demo.sh
./examples/run_roberta_graph_training_demo.sh
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
| `SEQ_LEN`      | `8`     | Passed to `--seq` (causal demos)             |
| `ENC_SEQ_LEN`  | `8`     | T5 encoder length (`--enc-seq`)                |
| `DEC_SEQ_LEN`  | `8`     | T5 decoder length (`--dec-seq`)                |
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

T5 seq2seq windows (encoder + decoder segments per batch):

```bash
python3 examples/prepare_tiny_seq2seq_train_bin.py \
    --output build/examples/demo_data/t5/train.bin \
    --enc-seq-len 8 --dec-seq-len 8 --batch-size 2 --num-batches 8
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

Tiling uses a separate **`tiling.json`** (axis keys match `config.json` plus `seq_len` / `batch_size` for `--seq` / `--batch`):

```json
{
  "default": {
    "batch_size": 1,
    "seq_len": [4, 4],
    "hidden_size": 32,
    "intermediate_size": [64, 64]
  },
  "layers": {
    "h_1": { "intermediate_size": [40, 88] }
  }
}
```

Demo configs: `examples/demo_configs/gpt2_tiny_config.json` and `gpt2_tiny_tiling.json`.

**Execution schedule** (`execution.json`): generate with round-robin, then reuse.

```bash
# 1) Generate execution.json (after compile, step 0)
./build/examples/gpt2_graph_training \
    --train-bin build/examples/demo_data/gpt2/train.bin \
    --config examples/demo_configs/gpt2_tiny_config.json \
    --tiling examples/demo_configs/gpt2_tiny_tiling.json \
    --execution-out /tmp/gpt2_execution.json \
    --seq 8 --batch 2 --epochs 1 --max-batches 1

# 2) Train with saved schedule
./build/examples/gpt2_graph_training \
    --train-bin build/examples/demo_data/gpt2/train.bin \
    --config examples/demo_configs/gpt2_tiny_config.json \
    --tiling examples/demo_configs/gpt2_tiny_tiling.json \
    --execution /tmp/gpt2_execution.json \
    --seq 8 --batch 2 --epochs 4 --max-batches 32
```

C++ API: `generate_round_robin_execution_schedule`, `write_execution_schedule_json`,
`load_execution_schedule_json`, `Runtime::set_execution_schedule`.

**Multi-GPU (single server):** pass `--ncuda N` (and optionally `--ncpu M`).
Regenerate `execution.json` when worker count changes. Full CUDA smoke is manual
(`--ncuda 2`, inspect `ops[].worker` in JSON); CPU-only builds use `--ncuda 0`.

**E2E script:** `run_gpt2_static_train.sh` runs generate-then-reload in one flow.

```bash
./build/examples/gpt2_graph_training \
    --train-bin build/examples/demo_data/gpt2/train.bin \
    --config examples/demo_configs/gpt2_tiny_config.json \
    --tiling examples/demo_configs/gpt2_tiny_tiling.json \
    --seq 8 --batch 2 --epochs 2 --max-batches 8
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

## GPT-Neo

| Artifact | Role |
|----------|------|
| `gptneo_graph_training` | Causal LM training on mmap `train.bin` (global + local attention masks) |
| `run_gptneo_graph_training_demo.sh` | Tiny data + multi-epoch demo |
| `gptneo_graph_training.cc` | Full CLI |
| `gptneo_generate` | Greedy generation (no KV cache; dual BOOL masks per step) |
| `gptneo_generate.py` | HF GPT-Neo → weights + prompt |

Example (manual training):

```bash
./build/examples/gptneo_graph_training \
    --train-bin build/examples/demo_data/gptneo/train.bin \
    --tiny --seq 8 --batch 2 --epochs 3 --max-batches 24 --lr 0.003
```

Example (generation):

```bash
python3 examples/gptneo_generate.py \
    --model EleutherAI/gpt-neo-125M \
    --output-dir /tmp/nntile_gptneo \
    --prompt "The meaning of life is"
./build/examples/gptneo_generate \
    --config /tmp/nntile_gptneo/config.json \
    --weights /tmp/nntile_gptneo/weights.safetensors \
    --prompt-ids "$(cat /tmp/nntile_gptneo/prompt_ids.txt)" \
    --max-tokens 32
```

Or let the binary invoke the Python converter:

```bash
./build/examples/gptneo_generate \
    --model EleutherAI/gpt-neo-125M \
    --prompt "Hello" \
    --max-tokens 16
```

## RoBERTa

| Artifact | Role |
|----------|------|
| `roberta_graph_training` | Tiny MLM training + save/load checkpoint (graph API) |
| `run_roberta_graph_training_demo.sh` | Runs training and prints scratch / best / loaded losses |
| `roberta_graph_training.cc` | Same flow as `bert_graph_training` (fixed toy batch) |
| `roberta_mlm_inference` | Forward-only MLM smoke test |

Example:

```bash
cmake --build build --target roberta_graph_training
./examples/run_roberta_graph_training_demo.sh
```

## T5

| Artifact | Role |
|----------|------|
| `t5_graph_training` | Seq2seq training on mmap `train.bin` (encoder-decoder, AdamW) |
| `run_t5_graph_training_demo.sh` | Tiny seq2seq data + multi-epoch demo |
| `t5_graph_training.cc` | Full CLI (`--tiny`, `--enc-seq`, `--dec-seq`, …) |
| `prepare_tiny_seq2seq_train_bin.py` | Offline `uint16` train.bin for demos |
| `t5_generate` | Autoregressive decoding (encoder + decoder steps) |
| `t5_generate.py` | HF T5 → weights + token ids |

Example (manual training):

```bash
./build/examples/t5_graph_training \
    --train-bin build/examples/demo_data/t5/train.bin \
    --tiny --enc-seq 8 --dec-seq 8 --batch 2 --epochs 3 \
    --max-batches 24 --lr 0.003
```

Example (generation):

```bash
python3 examples/t5_generate.py --model google/flan-t5-small \
    --output-dir /tmp/nntile_t5 \
    --encoder-prompt "translate English to German: Hello"
./build/examples/t5_generate \
    --config /tmp/nntile_t5/config.json \
    --weights /tmp/nntile_t5/weights.safetensors \
    --encoder-ids "$(cat /tmp/nntile_t5/encoder_ids.txt)" \
    --decoder-ids "$(cat /tmp/nntile_t5/decoder_ids.txt)" \
    --max-tokens 32
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
