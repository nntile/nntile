# Graph API: static tiling, task scheduling, GPT-2 training

**Status:** Final roadmap  
**Branch:** `graph_api`  
**Canonical copy:** `docs/dev/graph_static_execution_plan.md`

---

## Scope

Single-machine graph training with StarPU (CPU or CUDA workers).

| In scope | Out of scope |
|----------|----------------|
| `tiling.json` → static tile geometry | MPI / multi-node distribution |
| `execution.json` → **generate, save, load, run** | `home_node` / pinned tile memory |
| Explicit `generate_round_robin_execution_schedule()` | Automatic schedule inside `compile()` |
| Optional `execution.json` / `set_execution_schedule()` | Mandatory schedule for every `execute()` |
| Manual tiling + inspectable worker assignment | Autotuning |

**Data:** Tiles have no fixed StarPU home; the runtime may move copies. The schedule only states **which worker runs which op** and the **virtual** tile-to-worker split.

---

## Goals

| Goal | Done when |
|------|-----------|
| Graph GPT-2 trains | Loss decreases; incremental compile stable |
| Static tiling | `tiling.json` → `AxisDescriptor` |
| Static scheduling | `execution.json` drives `Runtime::execute()` |
| Round-robin generator | Explicit API + optional file round-trip |
| Multi-GPU (one server) | Loaded or generated schedule pins `STARPU_EXECUTE_ON_WORKER` |

---

## Artifacts

| File | Role |
|------|------|
| `config.json` | Model hyperparameters (input) |
| `tiling.json` | Tile sizes per axis / layer (input) |
| `execution.json` | Worker assignment ( **output** from generator, **input** to `execute()` ) |

### `execution.json` workflow

```text
  compile()  ──►  DCE execution order (no schedule yet)

  Path A — generate file (first time or retune)
    generate_round_robin_execution_schedule(graph, order)
    write_execution_schedule_json(schedule, "execution.json")

  Path B — reuse file (same graph structure / compile order)
    load_execution_schedule_json("execution.json")
    set_execution_schedule(schedule)

  execute()  ──►  per-op worker from schedule
```

**Rules (round-robin generator):**

1. Virtual tile `lin` → `worker = lin % num_workers`.
2. Single writable output → op runs on that tile’s worker.
3. Multiple writable / in-place outputs → worker with largest writable byte total (tie → lower id).

---

## API (implemented)

| Function | Purpose |
|----------|---------|
| `generate_round_robin_execution_schedule(graph, order)` | Build schedule in memory |
| `generate_round_robin_execution_json(graph, order, path)` | Generate + write file |
| `load_execution_schedule_json(path)` | Read file |
| `write_execution_schedule_json(schedule, path)` | Write file |
| `Runtime::compile()` | Lower + DCE only; **does not** set schedule |
| `Runtime::generate_round_robin_execution_schedule()` | After `compile()`, from internal order |
| `Runtime::set_execution_schedule(schedule)` | Optional: static worker pinning for `execute()` |
| `Runtime::load_execution_schedule(path)` | Load + set |
| `Runtime::compile_with_round_robin_schedule()` | Convenience: compile + in-memory round-robin |

---

## Architecture

```text
tiling.json ──► apply_flat_tiling_spec()
                      │
NNGraph record ──► finish_phase() ──► lower_and_compile() ──► Runtime::compile()
                      │
                      ├── generate_round_robin… ──► execution.json (out)
                      │
                      └── load execution.json ──► set_execution_schedule()
                      │
                      ▼
              bind_data → execute() → StarPU (worker per op)
```

---

## Current state

### Done

- Graph IR, incremental compile, GPT-2 `tiling.json`
- Round-robin generator + JSON load/write
- `Runtime` requires schedule before `execute()` (not embedded in `compile()`)
- `gpt2_graph_training`: `--execution`, `--execution-out`
- StarPU `STARPU_EXECUTE_ON_WORKER` when schedule is set

### Remaining work

1. **Tiling productization** — demo defaults, Python `--tiling`, move generic JSON helpers toward `include/nntile/tensor/`, CI tiled GPT-2 smoke test.

2. **Execution ergonomics** — document `execution.json` schema; optional regenerate policy on graph change (fingerprint); integration test on full GPT-2 block with tiling + load.

3. **Multi-GPU example** — `--ncuda` in `gpt2_graph_training`; README for generate → inspect → reload loop.

4. **More generators** — e.g. `generate_*_execution_schedule` as separate functions (same file format); still no schedule input flags beyond `--execution`.

5. **Production GPT-2** — Python config/checkpoints; heterogeneous tiling hardening (SDPA, embedding).

---

## Dependencies

```text
tiling.json + axis naming
        ↓
multi-tile lowering (attention, embedding, GEMM)
        ↓
execution.json generate / load + multi-GPU execute
        ↓
production GPT-2 tooling
```

**Main risk:** attention / embedding lowering under non-uniform tiles.

---

## GPT-2 reference commands

**Generate `execution.json` (first run):**

```bash
./build/examples/gpt2_graph_training \
  --train-bin build/examples/demo_data/gpt2/train.bin \
  --config examples/demo_configs/gpt2_tiny_config.json \
  --tiling examples/demo_configs/gpt2_tiny_tiling.json \
  --execution-out /tmp/gpt2_execution.json \
  --seq 8 --batch 2 --epochs 1 --max-batches 1
```

**Train using saved schedule:**

```bash
./build/examples/gpt2_graph_training \
  --train-bin build/examples/demo_data/gpt2/train.bin \
  --config examples/demo_configs/gpt2_tiny_config.json \
  --tiling examples/demo_configs/gpt2_tiny_tiling.json \
  --execution /tmp/gpt2_execution.json \
  --seq 8 --batch 2 --epochs 4 --max-batches 32
```

---

## Design principles

| Topic | Decision |
|-------|----------|
| Tiling | User `tiling.json` |
| Schedule | Explicit generator function; file is output **and** input |
| `compile()` | No implicit scheduling |
| Data | No `home_node`; single server only |
| StarPU | Worker from schedule; data movement stays dynamic |
| Incremental steps | Same tiling + same `execution.json` if graph order unchanged |

---

## Key source files

| Area | Path |
|------|------|
| Schedule API | `nntile/include/nntile/core/execution_schedule.hh` |
| Schedule impl | `nntile/src/core/execution_schedule.cc` |
| Runtime | `nntile/include/nntile/runtime.hh`, `nntile/src/runtime.cc` |
| StarPU hook | `nntile/include/nntile/starpu_c.hh` |
| Tiling | `nntile/examples/tiling_config_json.hh`, `gpt2_axis_naming.hh` |
| Example | `nntile/examples/gpt2_graph_training.cc` |
