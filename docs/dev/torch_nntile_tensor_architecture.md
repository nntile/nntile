# torch_nntile tensor architecture

Canonical description of `device=nntile` tensors as implemented on the
`graph_api` line (PR [#425](https://github.com/nntile/nntile/pull/425)).

## Model

A nntile tensor is a PyTorch `at::Tensor` shell (shape, dtype, autograd) with
**0-byte** `Storage`. Compute state lives in StarPU tiles behind a refcounted
`NodeRef` on the `TensorImpl`:

```text
TensorImpl → NNTileBackendMeta → NodeRef → NNTileBinding { logical L }
```

- **`L` (logical):** the graph/compute node. `NodeRef` ctor/dtor drive
  `mark_output(true/false)` on `L`.
- **`S` (staging):** **ephemeral**, not stored in `NNTileBinding`. Created per
  I/O event, invalidated after scatter/gather run.

There is no side map (`g_tensor_nodes`) and no eager per-op flush. All ops
record into a shared `TensorGraph`; the caller flushes with `compile_graph()`
+ `run()` (or legacy `execute()` = compile+run).

## I/O

| Direction | API | Behavior |
|-----------|-----|----------|
| Ingress | `.to("nntile")` / CPU→nntile `copy_` | Create `L`, attach `NodeRef`, create ephemeral single-tile `S`, write host bytes into `S`, record `scatter(S→L)`. Ingress is **once per tensor**; a second CPU copy into an already-bound tensor raises. After the scatter phase runs, ingress `S` is invalidated. |
| Egress | `.cpu()` / `.to("cpu")` / nntile→CPU `copy_` | Create ephemeral `S`, record `clear(S)` + `gather(L→S)`, **auto-compile and run** any pending ops plus the gather phase, StarPU-read `S`, invalidate `S`. |

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
  **share** the same `NodeRef` (no tile copy, no graph op). Non-contiguous
  `permute` and `Tensor.contiguous()` on non-contiguous nntile tensors error.
- At op-record time, a PyTorch shape that differs from `L`'s graph shape (same
  numel) may insert a `contiguous_view` **shape bridge**.
- **nntile→nntile `copy_`** with matching shape/dtype **aliases** `NodeRef`
  (same binding, no data copy). Distinct metadata tensors that cannot share a
  binding raise. There is no graph `tensor::copy` for this path.

## Autograd and norms

- Gradients use **PyTorch autograd**. LayerNorm / RMSNorm have dedicated
  backward paths.
- `torch.linalg.vector_norm` (ord=2) is **forward-only** on nntile. It raises if
  `input.requires_grad` and grad mode is enabled; under `torch.no_grad()` (or
  with a detached input) it is allowed. No CPU round-trips for unsupported
  `ord` / `dtype`.

## Optimizer steps

SGD / Adam / AdamW resolve the grad graph node from (in order):

1. `nntile_node(grad)` (binding on the grad tensor),
2. param-grad registry,
3. `lookup_data_node(grad)`.

They **do not** invent a fresh empty grad node. Missing registration raises;
run backward (or ingress a real grad tensor via `.to("nntile")`) first.

## Technical debt (future fixes)

| # | Topic | Current behavior | Planned follow-up |
|---|--------|------------------|-------------------|
| D1 | TensorGraph growth | Each `.cpu()` permanently appends `clear`, `gather`, and a new `io_staging_*` node. | Phase GC / compaction; reuse readout staging per session. |
| D2 | Incremental tile-map growth | Every ingress/egress lowers a fresh ephemeral `S` into `inc_state.tensor_to_tiles`; entries are never removed. | Reclaim staging descriptors after invalidate; or pool single-tile `S` per `L`. |
| D3 | Pin bookkeeping | `pin_tensor_for_graph` / ingress may append duplicate `at::Tensor` refs until the next graph clear. | Dedup by `TensorImpl*`; trim on phase seal. |
| D4 | CE `ignore_index` mean | Mean CE uses `1/numel`; PyTorch uses `1/count_non_ignore`. | Graph-native valid-label count (or document as permanent limitation). |
| D5 | `vector_norm` backward | Forward-only by design. | Add autograd when product needs it. |
| D6 | Stub vs libnntile | Builds without libnntile still use host `Storage` staging. | Keep stub path minimal; do not reintroduce host tiers on the libnntile path. |
