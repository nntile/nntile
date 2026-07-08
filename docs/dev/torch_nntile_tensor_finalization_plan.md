# Finalization plan: `device=nntile` PyTorch tensor

**Parent:** [nntile_tensor_impl_plan.md](nntile_tensor_impl_plan.md) (phases 0–7 largely complete)  
**Branch:** `cursor/pytorch-tensor-gc-investigation-94e3` / PR [#425](https://github.com/nntile/nntile/pull/425)  
**Audit date:** 2026-07-08

This document **re-states the target tensor contract**, reviews the current
`torch_nntile` implementation against it, and lists remaining work to **fully
converge** on the design.

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
| **Storage** | Always **0 bytes** under libnntile (`is_metadata_only_tensor` ≡ `nbytes()==0`). |
| **Views** | Same-numel `view`/`reshape`/`as_strided` (contiguous) / contiguous-preserving `permute` → **share `NodeRef`**, no graph op. |
| **Layout ops** | `transpose` / `.t()` → materialize (`swap_two_axes`). Non-contiguous `permute` → error. |
| **I/O in** | `.to("nntile")` / CPU→nntile `copy_`: new binding, `bind_data` on `S`, record `scatter(S→L)` at init. |
| **I/O out** | `.cpu()` / nntile→CPU `copy_`: record `gather(L→S)`, user `compile_graph()` + `run()`, read `S`, invalidate buffer. **No direct `L` read from PyTorch API.** |
| **Compile** | Caller-controlled `compile_graph()` + `run()`; **one `run()` per compile**. Growing `g_graph`; `seal_phase` only new ops. |
| **GC** | `is_output` = live `NodeRef` refcount (incl. autograd-retained tensors). No compile-time seal pass. DCE via op graph + marks. |

### 1.3 Allowed “helpful additions” (secondary, must not contradict core)

These are **recorder/session helpers**, not alternate tensor kinds:

| Mechanism | Purpose | Constraint |
|-----------|---------|------------|
| `pin_tensor_for_graph` / `g_pinned_tensors` | Keep live tensors visible across compile for axis naming / retention | Must not be required for correctness if `NodeRef` refcount is complete |
| `g_param_grad_registry` | Map param `TensorImpl` → grad `TensorNode` for accumulation / optimizer | Grad `at::Tensor` must still carry `NodeRef` |
| `g_axis_name_hints` / `g_axis_tiling_by_name` | Pre-compile axis metadata | Orthogonal to tensor identity |
| `g_label_host_cache` | INT64 label fast-path | **Violates** zero-host-bytes; should be removed or replaced with graph input |
| `ensure_graph_shape_bridge_locked` | Same-numel PyTorch shape ≠ graph shape | Should not insert `CONTIGUOUS_VIEW` for free-reshape cases |
| `TORCH_NNTILE_ASSERT_NODE_REF` | Debug invariant | Keep |

### 1.4 Explicit non-requirements (rejected)

- Host `std::vector` / dense bytes on `Storage`
- Three tiers (GraphHandle / Staged / Persistent)
- `is_staged_input_tensor`, `g_metadata_only_impls`, `needs_host_copy`
- Tile adoption, per-segment logical rebind, graph epoch id
- `seal_output_marks_*` at compile
- `aten::contiguous` materialization (error if not already contiguous)
- Optional host payload for grads / optimizer state

### 1.5 Public API invariants (testable)

1. Every libnntile tensor participating in a recorded op has `nntile_binding(t) != nullptr` after the op returns (when `TORCH_NNTILE_ASSERT_NODE_REF=1`).
2. `tensor.storage().nbytes() == 0` for all `device=nntile` tensors created via registered factories.
3. `view(x)` where `x.numel() == view.numel()` → `nntile_binding(x) == nntile_binding(view)` (same `shared_ptr`).
4. `.cpu()` after `compile`+`run` never calls `Runtime::get_output(L)` directly (only via `gather` → `S`).
5. `grad.zero_()` on metadata grad records `tensor::fill` in the graph (no `data_ptr` write).
6. No `tensor::contiguous_view` op recorded for pure reshape/view chains.

---

## 2. Implementation review (2026-07-08)

### 2.1 Aligned with design ✓

| Area | Location | Notes |
|------|----------|-------|
| `NodeRef` / `NNTileBinding` | `nntile_tensor_meta.{h,cpp}` | Sole graph link; ctor/dtor drives `mark_output` on `L` |
| 0-byte factory | `empty_metadata_tensor`, `empty.memory_format` | Registered in `nntile_kernels.cpp` |
| Free reshape (ATen) | `view`, `as_strided`, `permute` (contiguous) | `record_view_alias` → `share_node_ref_for_reshape` |
| Input path | `init_nntile_input_from_cpu` | scatter at init, `S` mark_input/output |
| Output path (main) | `copy_nntile_tensor_to_cpu` post-run | `gather_logical_to_staging_and_read_locked` |
| Incremental exec | `execute_range` in `RecorderExecState` | No full re-run of sealed phases |
| Tier removal | libnntile build | No `g_tensor_nodes`, no `ensure_host_staging` |
| Grad policy | `register_param_grad_node`, `register_grad_alias_for_host_copy`, `tensor_fill_fp32` | Multi-backward + `zero_` work |
| Compile seal removed | `compile_graph_locked` | No `seal_output_marks_*` |

### 2.2 Not yet within design ✗

Grouped by severity.

#### P0 — Correctness / design violation (libnntile)

| ID | File(s) | Issue |
|----|---------|-------|
| **F-01** | `nntile_kernels.cpp` `reshape_alias` | Registered as `_reshape_alias`; does **not** call `record_view_alias`. Autograd reshape can drop `NodeRef`. |
| **F-02** | `nntile_kernels.cpp` `set_source_tensor`, `set_source_storage*` | Rebinds `Storage`/strides with no `NodeRef` update. |
| **F-03** | `nntile_kernels.cpp` `resize_` | `resize_impl_cpu_` on 0-byte storage; no graph node update. |
| **F-04** | `nntile_graph_recorder.cpp` `read_logical_to_host_locked`, `read_nntile_logical_to_host` | Direct `Runtime::get_output(L)` — bypasses gather. Used by labels and staging populate. |
| **F-05** | `nntile_graph_recorder.cpp` `copy_nntile_tensor_to_cpu` L1330–1339 | Pre-first-compile staging fast-path skips gather (OK for fresh inputs only; document or narrow). |
| **F-06** | `nntile_kernels.cpp` `copy_from` | `can_read_nntile_tensor_from_staging` fast-path can export without gather. |
| **F-07** | `nntile_graph_recorder.cpp` `gather_logical_to_staging_and_read_locked` | **Inline** `compile_graph_locked` + `run_graph_locked` inside `.cpu()` — couples readout to implicit compile; inflates `mark_output` on `S` temporarily. |
| **F-08** | `nntile_graph_recorder.cpp` `ensure_graph_shape_bridge_locked` | Inserts `tensor::contiguous_view` when PyTorch shape vector ≠ graph shape but numel matches — breaks “free reshape” for permute/view shape reinterpretation. |
| **F-09** | `nntile_executor.cpp` `tensor_sum_to_scalar_fp32`, `nntile_broadcast.cpp` | 0-D `data_ptr` + `memcpy` on metadata tensors. |
| **F-10** | `nntile_executor.cpp` `labels_host_ptr` | Order: `g_label_host_cache` → `read_nntile_logical_to_host` → CPU copy. Host cache + logical bypass. |
| **F-11** | `nntile_norm.cpp` | `cpu_vector_norm_fallback` + `out.copy_` for several `linalg_vector_norm` cases — CPU compute, not graph. |
| **F-12** | `nntile_add.cpp` `broadcast_to_shape` | When `!has_pending_graph()`, CPU expand + `out.copy_(cpu_broadcast)` — no graph record. |
| **F-13** | *(missing)* `detach` | No `PrivateUse1` `detach` impl; may not share `NodeRef` on views. |

#### P1 — Consistency / technical debt

| ID | File(s) | Issue |
|----|---------|-------|
| **F-14** | `nntile_graph_recorder.cpp` | `g_pinned_tensors` parallel retention vs `NodeRef` refcount — dual liveness story. |
| **F-15** | `nntile_graph_recorder.cpp` | `g_label_host_cache` duplicates INT64 host bytes outside `S`. |
| **F-16** | `nntile_graph_recorder.cpp` | `populate_staging_from_logical_locked`, `refresh_input_scatter_locked` (dead) — logical-read gather bypass / dead code. |
| **F-17** | Op files (`nntile_linear.cpp`, `nntile_gemm.cpp`, …) | Widespread `at::empty` vs explicit `empty_metadata_tensor` — works via dispatch but inconsistent. |
| **F-18** | `nntile_sdpa_aten.cpp` | Placeholder `at::empty` debug tensors without `NodeRef`. |
| **F-19** | `nntile_generator.cpp` | RNG state uses host `data_ptr` on nntile storage. |
| **F-20** | `nntile_hooks.cpp` | Storage resize hooks assume host bytes. |

#### P2 — Stub-only / docs (no libnntile)

| ID | File(s) | Issue |
|----|---------|-------|
| **F-21** | `nntile_tensor_gc.cpp` | `ensure_host_staging`, `mark_staged_input_tensor`, `g_stub_staged_input_impls` — Phase-1 stub subsystem. |
| **F-22** | `#ifndef TORCH_NNTILE_USE_LIBNNTILE` branches | `nntile_kernels.cpp`, `nntile_add.cpp`, `nntile_sgd_step.cpp`, `nntile_cross_entropy.cpp`, … |
| **F-23** | `torch_nntile/README.md` L252–254 | Claims weights/inputs use “normal PyTorch host storage” — false for libnntile (0-byte + `S`). |
| **F-24** | `nntile_tensor_impl_plan.md` L285–290 | Still says gather missing / direct logical read — partially stale. |

---

## 3. Finalization plan

Phases **A–F** complete the tensor design after phases 0–7. Land on PR #425.

### Phase A — View / storage API hardening

**Goal:** Every ATen op that creates or mutates a tensor identity preserves or
explicitly breaks `NodeRef`.

| Task | Fix |
|------|-----|
| A.1 | `reshape_alias`: call `record_view_alias(self, result)` under libnntile (same as `view`). |
| A.2 | `detach`: add `PrivateUse1` impl → `share_node_ref_for_reshape` (no graph op). |
| A.3 | `set_.source_*`: `TORCH_CHECK(false, "... unsupported on nntile")` or propagate `NodeRef` if same binding policy. |
| A.4 | `resize_`: reject non-zero resize on metadata tensors; graph-aware path TBD or hard error. |

**Tests:** `test_view_shares_node_ref`, `test_reshape_alias_shares_node_ref`, `test_detach_shares_node_ref`, `test_set_raises_on_nntile`.

**Acceptance:** `TORCH_NNTILE_ASSERT_NODE_REF=1` passes on view/reshape/detach chains.

---

### Phase B — Eliminate logical-read and staging I/O bypasses

**Goal:** PyTorch host export always goes `L → gather → S → memcpy`; no
`Runtime::get_output(L)` from torch_nntile.

| Task | Fix |
|------|-----|
| B.1 | Delete or internalize `read_logical_to_host_locked` / `read_nntile_logical_to_host` from PyTorch API paths. |
| B.2 | `labels_host_ptr`: use `S` staging read or graph INT64 input node only; remove `g_label_host_cache` (F-15). |
| B.3 | Remove `populate_staging_from_logical_locked`; delete `refresh_input_scatter_locked` if unused. |
| B.4 | Narrow `copy_nntile_tensor_to_cpu` staging fast-path to **fresh inputs only** (`phase_seal_cursor==0` && never executed && scatter pending) or remove. |
| B.5 | Remove `can_read_nntile_tensor_from_staging` from `copy_from` fast-path. |

**Tests:** `test_cpu_always_gathers`, `test_no_logical_read_after_run`, CE with INT64 labels without host cache.

**Acceptance:** grep for `get_output` / `read_logical` under `torch_nntile/csrc` only in test helpers or deleted.

---

### Phase C — Decouple `.cpu()` from implicit compile

**Goal:** Readout records `gather(L→S)` into pending graph; **user** calls
`compile_graph()` + `run()` (or test helper `nntile_cpu`).

| Task | Fix |
|------|-----|
| C.1 | Split `gather_logical_to_staging_and_read_locked`: (1) `record_gather_to_staging(L,S)` only; (2) read `S` after user run. |
| C.2 | `copy_nntile_tensor_to_cpu`: if gather recorded and session not run → error with message (mirror `require_no_pending_graph` style) **or** document `nntile_cpu` as the supported read API. |
| C.3 | Stop temporary `staging->mark_output(true)` inflation during readout. |

**Decision needed:** Strict (always require explicit compile before `.cpu()`) vs
ergonomic (`nntile_cpu` / `.cpu()` auto-flush). **Recommend:** keep auto-flush
in `nntile_cpu` / `.cpu()` for UX but implement as “record gather + call shared
`execute_pending_if_needed`” without mutating `mark_output` on `S`.

**Tests:** `test_cpu_invalidates_staging_buffer`, `test_gather_recorded_before_run`.

---

### Phase D — Free reshape at graph level

**Goal:** Same-numel shape reinterpretation never inserts `CONTIGUOUS_VIEW`.

| Task | Fix |
|------|-----|
| D.1 | `lookup_data_node` / `get_or_create_data_node`: when numel matches and `NodeRef` shared, return `binding->logical` **without** `ensure_graph_shape_bridge_locked`. |
| D.2 | Reserve `ensure_graph_shape_bridge_locked` for true layout bridges (e.g. batched GEMM shape metadata), not PyTorch view shape vectors. |
| D.3 | Audit `permute` + matmul tests for spurious `CONTIGUOUS_VIEW` ops. |

**Tests:** `test_no_contiguous_view_op_on_reshape`, op-name assertion on view chain.

---

### Phase E — Metadata-safe scalar / broadcast / norm paths

**Goal:** No `data_ptr` / host memcpy on 0-byte tensors in libnntile paths.

| Task | Fix |
|------|-----|
| E.1 | `tensor_sum_to_scalar_fp32` 0-D branch: `tensor::copy` or identity node, not `memcpy`. |
| E.2 | `tensor_broadcast_scalar_fp32` 0-D: graph scalar node or `tensor::fill`. |
| E.3 | `nntile_norm.cpp`: route `linalg_vector_norm` fallback through graph ops or explicit `TORCH_CHECK` (no silent CPU copy). |
| E.4 | `broadcast_to_shape` without pending graph: record `repeat` or error — no CPU round-trip. |

**Tests:** extend `test_grad_zero_matches_cpu`; norm parity without CPU fallback path.

---

### Phase F — Retention model & cleanup

**Goal:** Single liveness story; stub retirement; docs match code.

| Task | Fix |
|------|-----|
| F.1 | Audit whether `g_pinned_tensors` can fold into `NodeRef` + `pin_hold` only; remove duplicate retention. |
| F.2 | Delete stub host-staging subsystem (`F-21`) or gate behind `!TORCH_NNTILE_USE_LIBNNTILE`-only build target. |
| F.3 | Standardize op outputs on `empty_metadata_tensor`. |
| F.4 | Update `torch_nntile/README.md`, `nntile_tensor_impl_plan.md` §3.6/§14, §11 inventory. |
| F.5 | Add §11 callsite inventory table (Phase 0 completion). |

**Acceptance:** README memory section describes 0-byte + `{L,S}` only; no “host storage on weights”.

---

## 4. Suggested execution order

```text
A (view API)  →  D (graph reshape)  →  B (I/O purity)  →  C (readout compile)
     ↓                                                              ↓
E (metadata scalars)                                      F (cleanup + docs)
```

- **A + D** are low-risk and unblock autograd shape ops.
- **B + C** are the largest design fidelity gap (logical read + implicit compile).
- **E** fixes latent footguns (0-D memcpy, norm fallback).
- **F** is ongoing cleanup; stub removal can be last.

---

## 5. Success criteria (final)

1. **Tensor = TensorNode provider:** `NodeRef` on every participating tensor; no side map; 0-byte `Storage`.
2. **I/O symmetry:** scatter-at-init, gather-on-export only; no `get_output(L)` from torch_nntile.
3. **Free reshape:** ATen + graph levels — no `CONTIGUOUS_VIEW` for same-numel reinterpretation.
4. **User-controlled compile:** readout does not silently seal extra phases or inflate `mark_output` (or documented auto-flush is intentional and tested).
5. **No host payload:** remove `g_label_host_cache`; no `data_ptr` writes on metadata tensors.
6. **Docs = code:** README and plan doc reflect libnntile reality.
7. **Tests:** core graph + grad + new invariant tests green; parity skips documented per-op.

---

## 6. Out of scope (separate tracks)

- RNG generator host state (`nntile_generator.cpp`)
- SDPA debug placeholders
- CUDA-only tests
- libnntile no-op scatter/gather when tiling matches (§3.7 future work)
- Broader parity test unskipping (softmax backward, embedding, etc.)
