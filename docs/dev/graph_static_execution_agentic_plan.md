# Agentic execution plan: graph static tiling + scheduling

**Parent roadmap:** [graph_static_execution_plan.md](graph_static_execution_plan.md)  
**Branch:** `graph_api`  
**Do not:** push to GitHub or open PRs unless the user asks.

This document turns the roadmap into **agent-sized tasks** with dependencies,
acceptance criteria, and verification commands. Agents should complete tasks in
order within each track; respect cross-track dependencies.

---

## Execution model

```text
Track A (tiling)     ──┐
Track B (execution)  ──┼──► Track D (E2E GPT-2) ──► Track E (production)
Track C (multi-GPU)  ──┘
```

| Role | Responsibility |
|------|----------------|
| **Implementer agent** | One task per session; run tests listed; update checklist |
| **Reviewer agent** | Diff vs acceptance criteria; no scope creep (no MPI, no DDP/FSDP) |
| **Integrator agent** | Track D/E only after A+B smoke green |

**Global invariants (every task):**

- Single server only; no `home_node`; no MPI distribution work.
- `compile()` must not auto-assign schedules (explicit generate/load only).
- `execution.json` is both generated and loadable; round-robin via named functions.
- No parallelism mode enums; no DDP/FSDP graph rewrites in this plan.

---

## Task checklist (master)

| ID | Task | Deps | Status |
|----|------|------|--------|
| A1 | Demo script: default `--tiling` + optional `--execution-out` | — | todo |
| A2 | Python `gpt2_training.py`: `--tiling`, `--execution`, `--execution-out` | pybind | todo |
| A3 | Refactor tiling JSON helpers to `include/nntile/tensor/` | — | todo |
| A4 | CI: tiled GPT-2 graph training smoke | A1 | todo |
| B1 | Document `execution.json` schema | — | todo |
| B2 | Schedule fingerprint / mismatch error on load | — | todo |
| B3 | Integration test: GPT-2 one block + tiling + execution load | A3 | todo |
| C1 | `gpt2_graph_training --ncuda` + Context wiring | — | todo |
| C2 | README: generate → inspect → reload loop | C1 | todo |
| C3 | Manual or CI test: 2 CUDA workers, loss decreases | C1,A1 | todo |
| D1 | E2E script: generate execution.json then train with `--execution` | A1,B1,C1 | todo |
| D2 | `docs/graph-wip.md` sync with agentic completion | D1 | todo |
| E1 | Python: full `config.json` load (not only `--tiny`) | A2 | todo |
| E2 | Checkpoint save/load for graph training | E1 | todo |
| E3 | Heterogeneous tiling tests: SDPA + embedding | B3 | todo |
| F1 | Second generator API (`affinity_batch` via `Runtime::generate_affinity_batch_execution_schedule`) | B1 | done |

Mark **done** in this table when merged to `graph_api`.

---

## Track A — Tiling productization

### A1 — Demo script defaults

**Goal:** `run_gpt2_graph_training_demo.sh` exercises tiled training by default.

**Files:**

- `nntile/examples/run_gpt2_graph_training_demo.sh`
- `nntile/examples/demo_configs/gpt2_tiny_tiling.json` (already exists)

**Steps:**

1. Pass `--config` and `--tiling` pointing at `demo_configs/`.
2. Optional env `EXECUTION_OUT` → `--execution-out` on first run.
3. Ensure `cmake --build build --target gpt2_graph_training` in script.

**Acceptance:**

- Script completes; loss summary shows decrease.
- Log mentions tiling path and optional execution-out.

**Verify:**

```bash
./nntile/examples/run_gpt2_graph_training_demo.sh
```

---

### A2 — Python graph GPT-2 parity

**Goal:** `python/examples/gpt2_training.py` matches C++ flags.

**Files:**

- `python/examples/gpt2_training.py`
- `python/nntile/_bindings/nntile.cc` (if bindings missing)

**Steps:**

1. Add CLI: `--tiling`, `--execution`, `--execution-out`.
2. After `lower_and_compile()`: load or generate+set schedule (mirror C++).
3. Document in `python/examples/README.md`.

**Acceptance:**

- `pytest python/tests` passes.
- One manual run with `--tiny` and tiling path.

**Verify:**

```bash
pytest python/tests -q
python3 python/examples/gpt2_training.py --train-bin ... --tiny \
  --tiling nntile/examples/demo_configs/gpt2_tiny_tiling.json \
  --execution-out /tmp/exec.json --max-batches 1
```

---

### A3 — Move tiling helpers to core/tensor

**Goal:** Reusable tiling JSON API outside `examples/`.

**Files:**

- Move/refactor from `nntile/examples/tiling_config_json.hh`
- New: `nntile/include/nntile/tensor/tiling_spec_json.hh` (+ `.cc` if needed)
- Keep `gpt2_axis_naming.hh` in examples (model-specific)
- Update `gpt2_graph_training.cc` includes
- Update `nntile/tests/tensor/tiling_config_json.cc`

**Acceptance:**

- Examples and tests build; behavior unchanged.

**Verify:**

```bash
cmake --build build --target gpt2_graph_training tests_graph_tensor_tiling_config_json
```

---

### A4 — CI smoke (tiled GPT-2)

**Goal:** CI runs graph GPT-2 demo or equivalent with `--tiling`.

**Files:**

- `.github/workflows/*.yml` (or existing build-test job)

**Steps:**

1. Add step after build: run demo script or minimal `gpt2_graph_training` invocation.
2. Timeout generous for CPU-only CI.

**Acceptance:**

- Green on `graph_api` push.

---

## Track B — Execution ergonomics

### B1 — `execution.json` schema doc

**Goal:** Stable contract for tools and agents.

**Files:**

- New: `docs/dev/execution_json_schema.md` (or section in `examples/README.md`)

**Content:**

- Top-level keys: `policy`, `hardware`, `virtual_tile_workers`, `ops[]`.
- Per-op: `index`, `op`, `name`, `worker`, `writable_tiles`, `read_tiles`.
- Workflow: generate → edit (optional) → load.
- Explicit: not a memory placement map.

**Acceptance:**

- Linked from `graph_static_execution_plan.md` and `graph-wip.md`.

---

### B2 — Schedule validation on load

**Goal:** Clear failure when `execution.json` does not match compiled graph.

**Files:**

- `nntile/src/runtime.cc` (`set_execution_schedule`)
- Optional: store `compile_fingerprint` in JSON on write

**Steps:**

1. On write, optional hash: op count + concatenated `op_name` list.
2. On load, compare; throw readable error if mismatch.
3. Document regeneration requirement when graph changes.

**Acceptance:**

- Unit test: load wrong op count → throws.
- Unit test: valid round-trip → execute succeeds.

**Verify:**

```bash
cmake --build build --target tests_graph_tile_execution_schedule
./build/nntile/tests/tests_graph_tile_execution_schedule
```

---

### B3 — GPT-2 block integration test

**Goal:** Tiled forward+backward on one block with execution load.

**Files:**

- `nntile/tests/model/gpt2/` or new `nntile/tests/tile/gpt2_block_schedule.cc`

**Steps:**

1. Build tiny graph: one `Gpt2Block` or minimal attention+MLP.
2. Apply small `FlatTilingSpec` or JSON fixture.
3. `compile()` → write execution.json → `load_execution_schedule()` → `execute()`.
4. Compare loss or tensor norm vs untiled reference (loose tolerance).

**Acceptance:**

- Test tagged `[gpt2][schedule]` passes locally.

---

## Track C — Multi-GPU (single server)

### C1 — `--ncuda` in GPT-2 example

**Goal:** User controls CUDA worker count from CLI.

**Files:**

- `nntile/examples/gpt2_graph_training.cc`
- Possibly `Context` construction (replace hardcoded `CONTEXT_NUM_CUDA = 0`)

**Steps:**

1. Add `--ncuda N` (default 0 or env `STARPU_NCUDA`).
2. Construct `Context` with `ncuda=N`.
3. Regenerated `execution.json` should show `num_workers == N` when CUDA available.

**Acceptance:**

- Builds with `USE_CUDA=ON`.
- Running with `--ncuda 2` does not crash; schedule lists 2 workers.

---

### C2 — README generate / reload loop

**Goal:** Copy-paste workflow for humans and agents.

**Files:**

- `nntile/examples/README.md`

**Content:**

1. Build with CUDA.
2. Run with `--execution-out` once.
3. Inspect `virtual_tile_workers` and `ops[].worker`.
4. Re-run with `--execution` only.

---

### C3 — Two-GPU smoke

**Goal:** Evidence that scheduling spreads work.

**Steps:**

1. Machine with 2+ GPUs (or skip in CI).
2. Run tiny training 4–8 steps with `--ncuda 2`.
3. Loss decreases; execution.json shows workers 0 and 1 used.

**Acceptance:**

- Record command + outcome in PR description or test log (manual OK).

---

## Track D — End-to-end closure

### D1 — Single orchestration script

**Goal:** One script runs full generate-then-train workflow.

**Files:**

- New: `nntile/examples/run_gpt2_static_train.sh`

**Steps:**

1. Build `gpt2_graph_training`.
2. Step 1: `--max-batches 1 --execution-out`.
3. Step 2: same args + `--execution` (no out).
4. Assert loss step 2 < step 1 (weak check).

**Acceptance:**

- Script exit 0 on CPU-only and CUDA if available.

---

### D2 — Documentation sync

**Goal:** All entry points point to the same story.

**Files:**

- `docs/graph-wip.md`
- `docs/dev/graph_static_execution_plan.md` (checklist → done)
- Root `README.md` link optional

---

## Track E — Production-shaped GPT-2 (later)

### E1 — Python config.json

Wire `load_gpt2_config_json` equivalent in Python; drop `--tiny`-only limitation.

### E2 — Checkpoints

Save/load weights + optimizer state after graph training step (C++ or Python).

### E3 — Heterogeneous tiling hardening

Extend tests where `lower_to_tile` throws today; fix or document constraints for SDPA/embedding.

---

## Track F — Additional generators (optional)

### F1 — Second generator function

**Goal:** Prove the pattern is pluggable.

**Files:**

- `execution_schedule.hh`: e.g. `generate_affinity_batch_execution_schedule()`
- Same JSON format; different `policy` string.

**Acceptance:**

- Test with 2-tile batch axis; different worker map than round-robin.
- `gpt2_graph_training` does not need new flag yet (API-only OK).

---

## Agent session template

When starting a task, the agent should:

1. Read `docs/dev/graph_static_execution_plan.md` (invariants).
2. Read this file; pick one **todo** task whose deps are **done**.
3. Load only files listed under that task.
4. Implement; run **Verify** commands.
5. Update the master checklist row to `done`.
6. Summarize: what changed, what to run next.

**Forbidden unless user asks:**

- `git push`, PR creation.
- MPI, multi-node, `home_node`, DDP/FSDP implementations.
- Auto-schedule inside `compile()`.
- Making `execution.json` input-only or output-only.

---

## Suggested agent order (minimal path to E2E)

```text
B1 ──► B2 ──► B3
A1 ──► A4
A3 ─────────────► B3
C1 ──► C2 ──► C3
        │
        └──► D1 ──► D2
A2 (parallel anytime after pybind check)
E* after D2
F1 optional
```

**First three tasks for a new agent run:** `A1`, `B1`, `B2` (parallel).

**Definition of “plan complete”:** All rows A1–D2 marked done; E* optional stretch.
