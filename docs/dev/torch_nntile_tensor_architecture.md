# torch_nntile tensor architecture

Canonical description of `device=nntile` tensors as implemented on the
`graph_api` line (PR [#425](https://github.com/nntile/nntile/pull/425)).

## Model

A nntile tensor is a PyTorch `at::Tensor` shell (shape, dtype, autograd) with
**0-byte** `Storage`. Compute state lives in StarPU tiles behind a refcounted
`TensorRef` on the `TensorImpl`:

```text
TensorImpl → NNTileBackendMeta → TensorRef → Hold → TensorNode (graph-owned)
```

- **`L` (logical):** the graph/compute `TensorNode`, owned by `TensorGraph`
  (`unique_ptr`). Ops hold raw `TensorNode*`.
- **`TensorRef`:** shared hold on `L`. Last hold drop records async
  `tensor::invalidate` into the graph. `TensorGraph::data()` returns a
  `TensorRef`; op factories that create temps use `emplace_data()` (no hold)
  so callers must `TensorRef::adopt` outputs they keep.
- **`S` (staging):** **ephemeral**, not stored on the meta. Created per I/O
  event; after scatter/gather `wait()`, StarPU tile buffers are dropped from
  the runtime map (not left live).

There is no side map (`g_tensor_nodes`) and no eager per-op flush. All ops
record into a shared `TensorGraph`; the caller flushes with `compile_graph()`
+ `run()` (or legacy `execute()` = compile+run). `run()` / `execute()` submit
asynchronously; call `wait()` to join StarPU. Host readout (`.to("cpu")`) also
waits. C++ `nntile::Runtime::execute()` is likewise submit-only; call
`Runtime::wait()`.

## Incremental session memory (`TensorRef` + `INVALIDATE`)

The session is **one incremental** `TensorGraph` / `TileGraph` (phases append;
do not reset the session to free memory). StarPU payload reclaim is an ordinary
async graph op:

1. Dropping a Python nntile tensor drops its `TensorRef` (last hold → append
   ordinary `tensor::invalidate` on `L` into the **current** TensorGraph).
   Persistent params / batches stay live via remaining refs on those tensors.
   `del` after the last use is recorded is therefore safe: invalidate is just
   another op after those uses; StarPU orders `invalidate_submit` after them.
2. On `compile_graph()`, for every tensor **touched by the unsealed phase**
   without a live `TensorRef`, append `tensor::INVALIDATE` if needed (O(phase);
   covers `emplace_data` temps that never held a `TensorRef`). Do **not**
   side-channel `invalidate_logical_tiles` before submit — that freed payloads
   before the phase’s consumer tasks were inserted (e.g. `del inputs` before
   compile → StarPU “handle is not initialized” on embedding).
   Then `seal_phase` + lower to `tile::INVALIDATE` → `invalidate_submit` +
   clear payload.
3. `run()` only submits the execution stream (compute + INVALIDATE). StarPU
   orders each invalidate after the handle’s last prior use. Only `wait()`
   joins StarPU.
4. Training loops must `del` the step loss (free autograd) **before**
   `compile_graph` so temps have no live `TensorRef` when INVALIDATE ops are
   selected. Inputs/labels may be `del`’d as soon as their last use is
   recorded; keep parameters and optimizer state referenced.

Do **not** call `gc.collect()` in the step loop — refcount drop from `del` is
enough. `train_full_batch_step` drops logits after the step runs.

## I/O

| Direction | API | Behavior |
|-----------|-----|----------|
| Ingress | `.to("nntile")` / CPU→nntile `copy_` | Create `L`, attach `TensorRef`, create ephemeral single-tile `S` on StarPU immediately, write host bytes into `S`, record `scatter(S→L)`. Ingress is **once per tensor**. Batched prefetch keeps every `S` until the scatter phase runs; `run()` executes each scatter, waits, and destroys that `S` before the next so StarPU's allocation cache can reuse the CUDA chunk for the next `L` (submitting all scatters then unregistering all `S` left cached buffers → settled ≈2×). |
| Egress | `.cpu()` / `.to("cpu")` / nntile→CPU `copy_` | Create ephemeral `S`, record `clear(S)` + `gather(L→S)`, **auto-compile and run** any pending ops plus the gather phase, StarPU-read `S`, fully release `S`. |

### `.cpu()` auto-flush (by design)

Host readout **may compile and run** the pending TensorGraph (and always runs
the gather phase). Callers do not need an explicit `compile_graph()`/`run()`
before `.cpu()` for correctness, but:

- Ordering relative to other pending work follows whatever is still recorded.
- Each `.cpu()` permanently appends gather/`io_staging_*` nodes to the session
  graph (see debt D1).

Test helper `nntile_cpu()` also flushes pending work before `.cpu()`.

## Views, reshape, and nntile→nntile `copy_`

- Same-numel `view` / contiguous `as_strided` / contiguous-preserving `permute`
  **share** the same `TensorRef` (no tile copy, no graph op). Non-contiguous
  `permute` and `Tensor.contiguous()` on non-contiguous nntile tensors error.
- At op-record time, a PyTorch shape that differs from `L`'s graph shape (same
  numel) may insert a `contiguous_view` **shape bridge**.
- **nntile→nntile `copy_`** with matching shape/dtype **aliases** `TensorRef`
  (same hold, no data copy). Distinct metadata tensors that cannot share a
  hold raise. There is no graph `tensor::copy` for this path.

## Autograd and norms

- Gradients use **PyTorch autograd**. LayerNorm / RMSNorm have dedicated
  backward paths.
- `torch.linalg.vector_norm` (ord=2) is **forward-only** on nntile. It raises if
  `input.requires_grad` and grad mode is enabled; under `torch.no_grad()` (or
  with a detached input) it is allowed. No CPU round-trips for unsupported
  `ord` / `dtype`.

## Optimizer steps

SGD / Adam / AdamW resolve the grad graph node from (in order):

1. `nntile_node(grad)` (`TensorRef` on the grad tensor),
2. param-grad registry,
3. `lookup_data_node(grad)`.

They **do not** invent a fresh empty grad node. Missing registration raises;
run backward (or ingress a real grad tensor via `.to("nntile")`) first.

## `STARPU_W`-only clears and async multi-step VRAM (D7)

Many kernels record a destination `tensor::clear` (StarPU `clear` codelet with
**only** `STARPU_W` on the handle) before an accumulating write. That clear has
**no** `STARPU_R` / `STARPU_RW` edge onto weights or other step-carried state.

Weight updates still serialize real compute across training steps (RAW/WAW on
parameter handles). The clears do **not**: as soon as the host
`compile_graph()`/`run()`-submits step \(N+1\) while step \(N\) is still in
flight, every clear of step \(N+1\) is already **ready**. StarPU therefore
allocates CUDA buffers for those destinations immediately — before the step’s
gemms become ready. Submitting \(N\) steps without a host sync therefore makes
VRAM jump to roughly **\(N\) activation/grad working sets at once** (observed
on `train_gpt2.py` when raising `--max-sequences` with `--batch-size 1`), not
grow gradually as the GPU finishes steps.

This is mostly a **StarPU scheduling heuristic** gap for neural nets: a smarter
graph scheduler would keep each `STARPU_W` clear next to the first
`STARPU_RW`/`STARPU_R` consumer producer of that tensor so the clear is not
runnable far ahead of the step that needs it. Until that exists:

- **Avoid pure `STARPU_W` dependencies** with no `STARPU_R` or `STARPU_RW` on
  the same handle in the same step’s ready cone when multi-step async submit
  is desired (prefer folding the zero into the first write, or attaching a
  fictitious / real read edge that ties the clear to prior step state).
- **Mitigate in examples** by syncing once per step after `optimizer.zero_grad`
  (so grad `INVALIDATE`s land in the **same** compile phase as the step) via
  host loss readout (`.to("cpu")`), not a bare `wait()`.

## Technical debt (future fixes)

| # | Topic | Current behavior | Planned follow-up |
|---|--------|------------------|-------------------|
| D1 | TensorGraph metadata growth | Each `.cpu()` permanently appends `clear`, `gather`, and a new `io_staging_*` node (op list grows). Phase outputs cleared after `wait()` are reclaimed via `pending_output_reclaim` (O(phase outputs), not a full tile-map scan). Historical TileGraph/TensorGraph nodes still accumulate in memory. | Phase GC / compaction; reuse readout staging per session. |
| D2 | Ingress `S` + StarPU alloc cache | Batched `.to("nntile")` keeps every ephemeral `S` until scatters run. Submitting all scatters before any `S` unregister leaves CUDA replicates of every `S` beside every `L`; unregister parks them in StarPU's allocation cache (`STARPU_USE_ALLOCATION_CACHE`) → settled ≈2×. `starpu_memchunk_tidy` only writebacks dirty chunks — it does **not** flush that cache. | During `run()`, execute each ingress scatter, `wait()`, then destroy that `S` before the next scatter so the next `L` reuses the cached chunk. |
| D4 | CE `ignore_index` mean | Mean CE uses `1/numel`; PyTorch uses `1/count_non_ignore`. | Graph-native valid-label count (or document as permanent limitation). |
| D5 | `vector_norm` backward | Forward-only by design. | Add autograd when product needs it. |
| D6 | libnntile required | `torch_nntile` always links `libnntile`; host-only stub installs are unsupported. | Keep the libnntile path the only storage/executor path; do not reintroduce a stub build. |
| D7 | `STARPU_W`-only clear vs async steps | Destination `clear` tasks use only `STARPU_W`, so they are ready as soon as submitted and allocate VRAM for every in-flight step at once under async multi-step `run()` (see section above). StarPU’s heuristics do not delay those clears until related `STARPU_RW` work. | Graph-aware scheduler that colocates clears with first real use; and/or stop emitting standalone `STARPU_W` clears (fold into first write / add a tying read). Examples sync per step via loss `.to("cpu")` after `zero_grad`. |
