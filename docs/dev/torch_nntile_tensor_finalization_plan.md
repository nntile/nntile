# Finalization plan: `device=nntile` PyTorch tensor

**Parent:** [nntile_tensor_impl_plan.md](nntile_tensor_impl_plan.md) (phases 0–7 largely complete)  
**Branch:** `cursor/pytorch-tensor-gc-investigation-94e3` / PR [#425](https://github.com/nntile/nntile/pull/425)  
**Audit date:** 2026-07-08  
**Revised:** 2026-07-08 (design review feedback)

This document **re-states the target tensor contract**, reviews the current
`torch_nntile` implementation against it, and lists remaining work to **fully
converge** on the design.

**Scope:** All items below are **`torch_nntile` extension** work unless explicitly
marked as libnntile. The extension adapts PyTorch to NNTile; it does not require
changing libnntile semantics for finalization.

---

## 1. Formal requirements: TensorNode provider

### 1.1 Role (one sentence)

> A `device=nntile` `at::Tensor` is a **PyTorch façade** whose sole authoritative
> compute identity is a refcounted link to `TensorGraph::TensorNode*` tiles;
> everything else is bookkeeping to integrate with ATen, autograd, and host I/O.

The tensor is **mainly a TensorNode provider**. PyTorch fields (`sizes`,
`strides`, `dtype`, autograd metadata) are the shell; **payload never lives on
`Storage`**.

### 1.2 Core object model

| Layer | Requirement |
|-------|-------------|
| **Identity** | `NNTileBackendMeta` on `TensorImpl` holds `NodeRef` → `NNTileBinding { L, S }` only. No side map (`g_tensor_nodes`), no tier enums. |
| **Logical (`L`)** | `binding->logical` — graph/compute node. `mark_output(true)` while any `NodeRef` exists; `mark_output(false)` in `NNTileBinding` destructor. |
| **Staging (`S`)** | `binding->io_staging` — single-tile I/O seam. Created at `.to("nntile")` or lazily before first `.cpu()`. `mark_output(false)`. Buffer invalidated after scatter-at-run and after each `.cpu()`; pointer kept. |
| **Storage** | Always **0 bytes** under libnntile (`is_metadata_only_tensor` ≡ `nbytes()==0`). **No resize, no `set_` on storage** — there is nothing to mutate. |
| **Views (ATen)** | Same-numel `view` / contiguous `as_strided` / contiguous-preserving `permute` → **share `NodeRef`** on the PyTorch shell (virtual reshape). |
| **Views (TensorGraph)** | NNTile **does not** treat same-numel reshape as a free tensor-graph op. When PyTorch shape at record time ≠ `L`'s graph shape but numel matches, the recorder may insert `tensor::contiguous_view` as a **shape bridge** — reshape is realized later at **tile / `nntile::core` level** during lowering. |
| **Layout ops** | `transpose` / `.t()` → materialize (`swap_two_axes`). Non-contiguous `permute` → error. |
| **I/O in** | `.to("nntile")` / CPU→nntile `copy_`: new binding, host bytes into `S`, record `scatter(S→L)` at init. |
| **I/O out** | `.cpu()` / nntile→CPU `copy_`: `gather(L→S)` + compile + run + read `S`. **No `Runtime::get_output(L)` from torch_nntile.** |
| **Compile** | Caller-controlled `compile_graph()` + `run()`; **one `run()` per compile**. |
| **GC** | `is_output` = live `NodeRef` refcount. No compile-time seal pass. |
| **No caching** | No host-side caches that bypass the graph/I/O path (e.g. label caches, logical-read shortcuts). Optimizations deferred until the base path is stable. |

### 1.3 Allowed “helpful additions” (secondary)

| Mechanism | Purpose | Constraint |
|-----------|---------|------------|
| `pin_tensor_for_graph` / `g_pinned_tensors` | Compile-time retention / axis naming | Extension session state |
| `g_param_grad_registry` | Param → grad `TensorNode` for accumulation / optimizer | Grad tensor still carries `NodeRef` |
| `g_axis_name_hints` / `g_axis_tiling_by_name` | Pre-compile axis metadata | Orthogonal to tensor identity |
| `ensure_graph_shape_bridge_locked` | PyTorch shape ≠ graph shape, same numel | **Intentional** — `contiguous_view` bridge at TensorGraph level |
| `TORCH_NNTILE_ASSERT_NODE_REF` | Debug invariant | Keep |

### 1.4 Explicit non-requirements (rejected)

- Host payload on `Storage`; `resize_`; `set_.source_*` mutating storage
- Host caches (`g_label_host_cache`, logical-read fast paths)
- `Runtime::get_output(L)` exposed from torch_nntile
- Three tiers, side maps, compile-time seal, tile adoption
- Same-numel **free reshape at TensorGraph level** (differs from ATen virtual reshape)

### 1.5 Public API invariants (testable)

1. `tensor.storage().nbytes() == 0` for libnntile `device=nntile` tensors from registered factories.
2. `view(x)` same numel → `nntile_binding(x) == nntile_binding(view)` (ATen virtual reshape).
3. Host export never calls `Runtime::get_output(L)` — only `gather` → `S` → host memcpy.
4. No host-side label or logical-read caches in the recorder.
5. `grad.zero_()` records `tensor::fill` in the graph (no `data_ptr` write on metadata tensors).

---

## 2. Design decisions from review (2026-07-08)

| Topic | Decision |
|-------|----------|
| **F-01 `reshape_alias`** | **OK as-is.** `view` calls `record_view_alias` after `reshape_alias`; a separate `_reshape_alias` hook without alias is acceptable. |
| **F-02/03 `set_` / `resize_`** | **Non-issue / out of scope.** There is no `Storage` payload; resize is not supported. No plan work unless we add explicit `TORCH_CHECK` rejects later. |
| **F-04/10 logical read + label cache** | **Remove.** No `Runtime::get_output(L)` in torch_nntile; no `g_label_host_cache` or similar caches at this time — optimizations risk breaking invariants. |
| **F-06/07 copy / gather readout** | **torch_nntile extension** items only — not libnntile changes. Recorder/I/O code in `nntile_graph_recorder.cpp` / `nntile_kernels.cpp`. |
| **F-08 `contiguous_view` bridge** | **Correct by design.** TensorGraph keeps canonical `L` shape; PyTorch may present a different same-numel shape; bridge op connects them; true reshape happens at tile/core lowering. |

### Two-layer reshape model (canonical)

```text
ATen (PyTorch shell)     share NodeRef, new sizes/strides — virtual, no graph op
        ↓ record op with tensor.sizes() at call site
TensorGraph              L node shape may differ → contiguous_view bridge if same numel
        ↓ lower_to_tile
Tile / nntile::core      physical layout / reshape
```

---

## 3. Implementation review (revised)

### 3.1 Aligned with design ✓

| Area | Location |
|------|----------|
| `NodeRef` / `NNTileBinding` | `nntile_tensor_meta.{h,cpp}` |
| 0-byte storage | `empty_metadata_tensor`, factories |
| ATen virtual reshape | `view`, `as_strided`, contiguous `permute` |
| Graph shape bridge | `ensure_graph_shape_bridge_locked` + `contiguous_view` |
| Input scatter at init | `init_nntile_input_from_cpu` |
| Gather readout (main path) | `gather_logical_to_staging_and_read_locked` |
| Incremental `execute_range` | `RecorderExecState` |
| Grad `NodeRef` + graph `fill_`/`zero_` | executor + recorder |

### 3.2 Remaining gaps (re-prioritized)

#### P0 — torch_nntile extension (agreed must-do)

| ID | File(s) | Issue | Action |
|----|---------|-------|--------|
| **F-04** | `nntile_graph_recorder.cpp` | `read_logical_to_host_locked`, `read_nntile_logical_to_host` call `Runtime::get_output(L)` | **Delete** from torch_nntile; route all host reads through `S` after gather |
| **F-10** | `nntile_graph_recorder.cpp`, `nntile_executor.cpp` | `g_label_host_cache` + logical read in `labels_host_ptr` | **Delete cache**; labels via `S` / graph input only |
| **F-16** | `nntile_graph_recorder.cpp` | `populate_staging_from_logical_locked` uses logical read | **Delete** or rewrite via gather |
| **F-15** | (same as F-10) | Host INT64 bytes outside `S` | **Delete** |

#### P1 — torch_nntile extension (I/O recorder hygiene)

| ID | File(s) | Issue | Action |
|----|---------|-------|--------|
| **F-05** | `copy_nntile_tensor_to_cpu` | Pre-first-compile staging read skips gather | Document as input-only fast path, or remove for uniformity |
| **F-06** | `nntile_kernels.cpp` `copy_from` | `can_read_nntile_tensor_from_staging` bypass | Align with gather policy or restrict to fresh inputs |
| **F-07** | `gather_logical_to_staging_and_read_locked` | Inline `compile_graph` + `run` inside `.cpu()` | Extension policy choice: keep ergonomic auto-flush or require explicit compile; **not** a libnntile change |
| **F-09** | `tensor_sum_to_scalar_fp32`, `tensor_broadcast_scalar_fp32` | 0-D `data_ptr` memcpy on metadata tensors | Graph scalar path or hard error |
| **F-11** | `nntile_norm.cpp` | CPU fallback + `out.copy_` | Graph ops or explicit unsupported error |
| **F-12** | `nntile_add.cpp` `broadcast_to_shape` | CPU round-trip when no pending graph | Graph `repeat` or error |

#### P2 — Accepted / no action

| ID | Notes |
|----|-------|
| **F-01** | `reshape_alias` without `record_view_alias` — OK |
| **F-02** | `set_.source_*` — no storage to mutate; out of scope |
| **F-03** | `resize_` — not supported; out of scope |
| **F-08** | `contiguous_view` shape bridge — **intentional** |
| **F-13** | `detach` — optional; only if autograd path proves broken |

#### P3 — Cleanup / docs / stub

| ID | Issue |
|----|-------|
| **F-14** | `g_pinned_tensors` vs `NodeRef` — audit, simplify if possible |
| **F-17–F-20** | Style consistency, SDPA placeholders, RNG, hooks |
| **F-21–F-24** | Stub host-staging subsystem; README/plan doc staleness |

---

## 4. Finalization plan (revised phases)

All work is in **`torch_nntile/`** (primarily `csrc/nntile_graph_recorder.cpp`,
`nntile_kernels.cpp`, `nntile_executor.cpp`).

### Phase A — Remove logical read and all host caching (P0)

**Status:** **Done** (2026-07-08)

**Goal:** torch_nntile never calls `Runtime::get_output(L)`; no recorder caches.

| Task | Action |
|------|--------|
| A.1 | Remove `read_logical_to_host_locked` / `read_nntile_logical_to_host` and all callers. |
| A.2 | Remove `g_label_host_cache`; stop populating it in `init_nntile_input_from_cpu`. |
| A.3 | Rewrite `labels_host_ptr` to read INT64 labels from `S` (post-scatter / post-run) or from the graph operand node via staging — **no cache**. |
| A.4 | Delete `populate_staging_from_logical_locked`; delete dead `refresh_input_scatter_locked`. |

**Tests:** cross-entropy with INT64 labels; grep confirms no `get_output` on logical nodes in torch_nntile.

**Acceptance:** No host-side caches; no direct logical read API in extension.

---

### Phase B — torch_nntile I/O path consistency (P1)

**Status:** **Partial** — `copy_from` staging fast-path removed; gather readout policy unchanged (ergonomic inline compile on `.cpu()`).

**Goal:** Single story for CPU export in the extension recorder.

| Task | Action |
|------|--------|
| B.1 | Audit `copy_nntile_tensor_to_cpu` and `copy_from` staging fast-paths — document or remove. |
| B.2 | Policy for `gather_logical_to_staging_and_read_locked` inline compile+run: keep (ergonomic `.cpu()`) or split record vs execute — **extension-only** decision. |
| B.3 | Ensure every post-run host read goes `gather(L→S)` → run → read `S` → invalidate `S`. Input `S` stays readable after scatter until `.cpu()` gather readout. |

**Tests:** `test_cpu_invalidates_staging_buffer`, `test_nntile_to_cpu_copy`, graph execution training loop.

---

### Phase C — Document two-layer reshape model (P2 → doc)

**Status:** **Done** — README + plan updated.

**Goal:** Code and docs match the TensorGraph vs ATen reshape split.

| Task | Action |
|------|--------|
| C.1 | Document `ensure_graph_shape_bridge_locked` + `contiguous_view` in README and plan doc. |
| C.2 | Clarify ATen `share_node_ref` (virtual) vs graph bridge (canonical `L` shape). |
| C.3 | No code change to remove `contiguous_view` bridge. |

---

### Phase D — Metadata-safe scalars and CPU fallbacks (P1, optional)

**Status:** **Partial** — 0-D `tensor_sum_to_scalar_fp32` and `tensor_broadcast_scalar_fp32` use graph `copy`; norm/broadcast CPU fallbacks remain.

| Task | Action |
|------|--------|
| D.1 | Fix 0-D `data_ptr` paths in `tensor_sum_to_scalar_fp32`, `tensor_broadcast_scalar_fp32`. |
| D.2 | Replace or reject `linalg_vector_norm` CPU fallback and `broadcast_to_shape` CPU path. |

---

### Phase E — Cleanup and stub retirement (P3)

| Task | Action |
|------|--------|
| E.1 | Update `torch_nntile/README.md` (0-byte storage, no host cache, shape bridge). |
| E.2 | Sync `nntile_tensor_impl_plan.md` §3.2 reshape wording with two-layer model. |
| E.3 | Retire `#ifndef TORCH_NNTILE_USE_LIBNNTILE` host-staging when stub build deprecated. |
| E.4 | Complete §11 callsite inventory (Phase 0). |

---

## 5. Suggested execution order

```text
A (remove logical read + caches)  →  B (I/O consistency)  →  C (docs)
                                        ↓
                                   D (scalars/fallbacks, optional)
                                        ↓
                                   E (cleanup)
```

**Start with Phase A** — highest agreement, removes fragile optimizations.

---

## 6. Success criteria (revised)

1. **Tensor = TensorNode provider** — `NodeRef` on `TensorImpl`; 0-byte `Storage`; no side map.
2. **No direct logical read** — torch_nntile does not call `Runtime::get_output(L)`.
3. **No host caches** — no `g_label_host_cache` or equivalent in recorder.
4. **I/O via `S` only** — scatter in, gather out (extension recorder).
5. **Two-layer reshape documented** — ATen virtual reshape + TensorGraph `contiguous_view` bridge where needed; tile/core handles physical layout.
6. **Extension scope only** — finalization does not require libnntile semantic changes.
7. **Tests** — core graph + grad + CE labels green after Phase A.

---

## 7. Out of scope

- `reshape_alias` / `set_` / `resize_` changes (accepted or unsupported)
- Removing `contiguous_view` graph bridge (intentional)
- libnntile no-op scatter/gather (§3.7 future)
- RNG generator, SDPA debug tensors
- Broader parity unskipping (softmax backward, etc.)
