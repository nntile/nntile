# Adding a non-tiled torch kernel as a StarPU codelet

**Status:** design for `graph_api_torch_kernels`  
**Scope:** single-tile (untiled) tensors on `device=nntile` only  
**Related:** [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md),
[../cpp/README.md](../cpp/README.md), [../graph.md](../graph.md)

This note describes how to plug a **libtorch / ATen** kernel into the
Graph API stack so that an untiled `device=nntile` op still runs under
StarPU’s task runtime, without a matching `nntile::kernel` implementation.

CPU and CUDA StarPU workers are both in scope: the same aten schema runs
on `device=CPU` or `device=CUDA` blobs; CUDA must use the **StarPU** stream
and cuBLAS handle (never streams/handles from the `from_blob` tensor).

## Invariant (torch-native path)

Under `NNTILE_TORCH_NATIVE_OPS`, TensorGraph **compute** ops are **only**
torch-native aten ops (`TorchKind` / `TensorTorch*Op`). Classic NNTile
kernels (`swap_two_axes`, `scale_slice`, gemm codelets, …) are **not**
TensorGraph compute ops on this path.

Each such op must lower to the **same aten schema** inside the StarPU
codelet:

1. `at::from_blob` on **`device=CPU`** or **`device=CUDA`** (StarPU-owned
   buffers; empty deleter). The tensor is **meta + pointer only**.
2. Call the matching `at::*_out` / `*_copy_out` / functional aten API.
3. Wrap with `at::NoGradGuard` and
   `at::AutoDispatchBelowADInplaceOrView` so execution does not re-enter
   PrivateUse1 or Autograd.
4. **CUDA only:** bind ATen to the StarPU worker via `TorchCudaEnv`
   (see [CUDA StarPU workers](#cuda-starpu-workers) below).

I/O ops (`fill` / `subcopy` / scatter / gather / …) may stay classic.

`TorchKind` names follow aten (e.g. `TransposeCopy`, `NarrowCopy`,
`NativeLayerNorm`), not NNTile classic names.

## Goal

```text
torch / autograd
    → PrivateUse1 aten impl (device=nntile)
    → TensorGraph OpNode  (torch-native only; record)
    → lower (single tile only) → TileGraph OpNode
    → Runtime execute → nntile::core (torch-based)
    → nntile::starpu::submit → StarPU codelet
    → CPU:  from_blob(CPU)  + same aten::*_out (no grad)
    → CUDA: TorchCudaEnv (StarPU stream + cuBLAS)
            + from_blob(CUDA) + same aten::*_out (no grad)
```

Tiling remains the product path for `nntile::kernel` ops later; for this
torch-kernel path, **multi-tile graphs must be rejected**.

## Layer responsibilities

### 1. PrivateUse1 (torch_nntile)

User (or autograd) calls a native aten op. The PrivateUse1 impl:

1. Validates dtypes / contiguity / devices as today.
2. Uses **`device=meta`** when output shape/strides are non-obvious or may
   change across Torch versions:
   - Build meta clones of inputs (`empty_like(..., device=kMeta)` or
     equivalent).
   - Call the **functional** aten op on meta tensors.
   - Read sizes / dtype / strides from the meta result.
3. Allocates the nntile output with `empty` / `empty_like` from that meta
   result (or a trivial shape rule when the op is obvious, e.g. same-shape
   unary).
4. Registers the compute into **TensorGraph** (does not run StarPU yet).

Prefer driving compute through **`*_out`** variants conceptually: functional
forms allocate then share the same structured `impl` as `add.out` / `mm.out`.
Inside the StarPU codelet, call the out-style API with preallocated buffers.

**Before** registering on PrivateUse1, check how stock PyTorch registers the
same schema for CUDA — see
[Match device=cuda registration](#match-devicecuda-registration). Overriding a
`CompositeImplicitAutograd` op (e.g. `linear`, `chunk`) shadows CUDA’s
decomposition and often breaks Autograd.

Register functional, inplace, and out overloads separately on PrivateUse1
**only when CUDA has a device kernel** for that schema (or nntile storage
requires a hook — `as_strided` / `alias`). Out/inplace alone do **not** make
functional `add` work on PrivateUse1.

### 2. TensorGraph / TensorNode metadata

Every `TensorNode` must store **tensor meta**, not only shape:

| Field | Purpose |
|-------|---------|
| shape | extents (already present) |
| dtype | already present |
| strides | row-major / explicit strides for `from_blob` |
| memory format / contiguity | match ATen expectations |
| scalar type / options bits as needed | rebuild `TensorOptions` |

Record the op as a TensorGraph `OpNode` that knows it is **torch-native**
(distinct from classic `nntile::kernel` ops). Inputs/outputs are
`TensorNode` pointers carrying that meta.

### 3. Lowering TensorGraph → TileGraph (enforced single tile)

`OpNode::lower_to_tile`:

1. Resolve each operand’s tile list via the session tiling / layout.
2. **Require exactly one tile per operand** (`tiles.size() == 1` and
   `grid_volume == 1`). Otherwise throw (same policy as the temporary
   global tiling disable).
3. Emit **one** TileGraph operation that references those single
   `TileNode`s.

### 4. TileGraph / TileNode metadata

Each `TileNode` must carry the **same tensor meta** as its logical
`TensorNode` (for a single tile, tile shape equals tensor shape; strides
match the buffer layout StarPU will hold).

At execute time, the tile op calls a dedicated **`nntile::core`** entry that
is clearly named as torch-based (e.g. `core::torch_binary_out`), **not** a
`nntile::kernel::*` function.

### 5. `nntile::core` (torch-based, no `nntile::kernel`)

Signature sketch:

```cpp
// Torch-based core path: Tile + ATen meta → StarPU submit.
// No nntile::kernel counterpart.
void torch_binary_out(
    int starpu_worker_hint,
    starpu::TorchKind kind,  // e.g. TorchKind::Add
    const Tile<fp32_t> &a,
    const TorchTileMeta &a_meta,
    const Tile<fp32_t> &b,
    const TorchTileMeta &b_meta,
    const Tile<fp32_t> &out,
    const TorchTileMeta &out_meta,
    const starpu::TorchDispatchArgs &extra = {});
```

`TorchTileMeta` holds sizes, strides, and `storage_offset` (elements) needed
to rebuild `at::Tensor` views via `from_blob`. For `TorchKind::Add`, put
torch alpha in `extra.scalars[0]`.
Core only packs handles and calls `nntile::starpu::torch_binary.submit(...)`.

### 6. `nntile::starpu` codelet

Follow the existing codelet pattern (`Codelet` + `args_t` + `submit` +
CPU wrapper), with these differences:

1. **`submit`** compresses **all** input/output tensor meta into a single
   `args_t` (type unique per codelet) and passes it to
   `starpu_task_insert` / `nntile_starpu_task_insert` together with StarPU
   data handles for the tile buffers.
2. **Access modes (required):** every tensor argument must be classified
   as read-only, write-only, read-write, or workspace, and submitted with
   the matching StarPU mode (see [Data access modes](#data-access-modes)
   below). Never submit the same handle twice as `STARPU_R` + `STARPU_W`;
   use `STARPU_RW` once instead.
3. **CPU / CUDA function** (`void (*)(void *buffers[], void *cl_args)`):
   - Unpack `args_t`.
   - For each buffer, build `at::Tensor` with `torch_blob::blob_*` /
     `at::from_blob` using the raw pointer and the meta from `args_t`.
   - Use an **empty deleter** so destroying the temporary `Tensor` does
     **not** free StarPU-owned memory.
   - Call the low-level aten API, preferably `at::<op>_out(...)` /
     `at::_ops::<op>_out::call(...)`.
   - Wrap with `NoGradGuard` + `AutoDispatchBelowADInplaceOrView`.
4. **CUDA:** register `cuda_funcs[0]`; body starts with `TorchCudaEnv`
   (StarPU stream + cuBLAS). Keep the default `STARPU_CUDA_ASYNC` flag
   (same as classic kernels: enqueue on the StarPU worker stream and
   return). Do **not** hard-`restrict_where(STARPU_CPU)` on the codelet —
   `Context::restrict_cpu` / `restrict_cuda` set affinity.

## CUDA StarPU workers

Torch-native compute codelets support CUDA when NNTile is built with
`USE_CUDA=ON`. All binding lives under `nntile::starpu` only.

### `TorchCudaEnv` (`torch_cuda_env.hh`)

RAII helper used at the start of every CUDA codelet body:

1. `starpu_cuda_get_local_stream()` — worker’s CUDA stream
2. `starpu_cublas_get_local_handle()` + `cublasSetStream(handle, stream)`
3. Device id from `starpu_worker_get_devid(starpu_worker_get_id())`
4. `at::cuda::CUDAGuard` + `CUDAStreamGuard` with
   `at::cuda::getStreamFromExternal(stream, device_index)` so ATen
   kernels enqueue on the **StarPU** stream
5. Sets `torch_blob::default_device_tls()` to `cuda:devid` for the
   duration of the guard (restored on destroy)

**Never** take the stream or cuBLAS handle from a `from_blob` tensor —
those tensors are meta + StarPU pointer only.

StarPU’s cuBLAS handle is bound to that stream. ATen BLAS entry points
still use PyTorch’s own handle, but under `CUDAStreamGuard` they run on
the same StarPU stream.

### Codelet recipe

```cpp
void cpu(void *buffers[], void *cl_args) noexcept
{
    // from_blob via TLS (CPU by default) + at::*_out
}

#ifdef NNTILE_USE_CUDA
void cuda(void *buffers[], void *cl_args) noexcept
{
    TorchCudaEnv cuda_env; // StarPU stream + cuBLAS; TLS → CUDA
    (void)cuda_env;
    cpu(buffers, cl_args); // same aten body; blobs on CUDA
}
#endif
```

Generic Unary / Binary / Ternary may pass `cuda_env.device()` explicitly
into `run_*` instead of relying on TLS; specialized codelets prefer TLS +
shared `cpu()` body.

Register both functions:

```cpp
codelet("nntile_torch_op", footprint, cpu_funcs, cuda_funcs);
```

CUDA impls stay `STARPU_CUDA_ASYNC` like classic NNTile kernels: bind
ATen to StarPU’s worker stream via `TorchCudaEnv` and return after
enqueue.

If an `aten::*_out` schema is **CPU-only** (e.g. `narrow_copy.out`,
`transpose_copy.out`), implement the same semantics with a view +
`copy_` so the StarPU CUDA worker path stays valid.

`_scaled_dot_product_flash_attention_for_cpu` / `_…_for_cpu_backward`
are also CPU-only. On CUDA workers, `torch_sdpa_backward` prefers
`_scaled_dot_product_efficient_attention` (+ backward; fp32-capable,
no full S×S materialize) and falls back to an explicit math SDPA
backward (`matmul` + `softmax`) under `NoGradGuard`. Pass
`is_causal=true` through the graph when possible instead of a dense
causal mask — materializing `[S,S]` bool/float masks dominates at
long sequence lengths.
### Context affinity

| API | Torch-native compute codelets |
|-----|-------------------------------|
| `restrict_cpu()` | `STARPU_CPU` |
| `restrict_cuda()` | `STARPU_CUDA` |
| `restore_where()` | clear restriction |

### Still out of scope

- Autograd / `requires_grad` inside codelets
- Replacing I/O kernels (`fill`, `subcopy`, …) with Torch (classic CUDA
  I/O may still be compiled when `USE_CUDA` is on)
- Sharing one StarPU codelet across many ATen ops long-term (acceptable as
  bootstrap for Unary/Binary/Ternary; prefer named per-op codelets for
  profiling)

## Data access modes

Every torch-native StarPU task must declare how each buffer is used.
NNTile logical roles map to StarPU modes as follows:

| Role | Meaning | StarPU mode |
|------|---------|-------------|
| read-only | Task reads; must not write | `STARPU_R` |
| write-only | Task fully overwrites; prior value unused | `STARPU_W` |
| read-write | Task reads then updates the same buffer | `STARPU_RW` |
| workspace | Temporary scratch; no live value across tasks | `STARPU_SCRATCH` |

Rules:

1. Prefer out-of-place `*_out` shapes: all inputs `STARPU_R`, each
   distinct output `STARPU_W`.
2. If the same StarPU handle is both an input and the destination
   (accumulate / true in-place), submit it **once** as `STARPU_RW`.
3. Use `STARPU_SCRATCH` only for StarPU-managed temporary tiles. Host
   allocations inside ATen (not registered with StarPU) are not
   scratch handles—document them as “ATen-internal temp” instead.
4. PrivateUse1 “inplace” APIs (`add_`, `silu_`, …) may still lower to
   **new** TensorGraph nodes (SSA). In that case StarPU still sees
   distinct `R` + `W` handles; that is not `STARPU_RW` unless the tile
   handle is actually shared.

### Implemented ops (current)

Family codelet `torch_unary` (one `R` input, one `W` output):

| `TorchKind` | Aten | Handles |
|-------------|------|---------|
| `MulScalar` | `mul.Scalar_out` | in `R`, out `W` |
| `Relu` | `relu.out` | in `R`, out `W` |
| `Silu` | `silu.out` | in `R`, out `W` |
| `Gelu` | `gelu.out` | in `R`, out `W` |
| `Softmax` | `_softmax.out` | in `R`, out `W` |
| `LogSoftmax` | `_log_softmax.out` | in `R`, out `W` |
| `Sum` | `sum.IntList_out` | in `R`, out `W` |
| `VectorNorm` | `linalg_vector_norm.out` | in `R`, out `W` |
| `NarrowCopy` | `narrow` view + `copy_` | in `R`, out `W` |
| `Repeat` | `repeat.out` | in `R`, out `W` |
| `AvgPool2d` | `avg_pool2d.out` | in `R`, out `W` |
| `AdaptiveAvgPool2d` | `_adaptive_avg_pool2d.out` | in `R`, out `W` |

`Repeat` stores factors for the **output** rank (`iargs[0..out_ndim)`).
Bias broadcast for `addmm` / `linear` may reshape a 1D bias to `[1, N]`
(view of the parent storage tile). Record paths must
`pack_tensor_layout` so execute-time `from_blob` rebuilds that view;
the CPU/CUDA wrapper must not take the repeat count from `in_ndim`
alone (parent tile rank can stay 1).
| `TransposeCopy` | `transpose` view + `copy_` | in `R`, out `W` |

Family codelet `torch_binary` (two `R` inputs, one `W` output):

| `TorchKind` | Aten | Handles |
|-------------|------|---------|
| `Add` | `add.out` (alpha in `scalars[0]`) | a `R`, b `R`, out `W` |
| `Mul` | `mul.out` | a `R`, b `R`, out `W` |
| `Hypot` | `hypot.out` | a `R`, b `R`, out `W` |
| `ThresholdBackward` | `threshold_backward` | grad_out `R`, self `R`, grad_in `W` |
| `SiluBackward` | `silu_backward` | grad_out `R`, self `R`, grad_in `W` |
| `GeluBackward` | `gelu_backward` | grad_out `R`, self `R`, grad_in `W` |
| `SoftmaxBackward` | `_softmax_backward_data` | grad_out `R`, output `R`, grad_in `W` |
| `LogSoftmaxBackward` | `_log_softmax_backward_data` | grad_out `R`, output `R`, grad_in `W` |
| `AvgPool2dBackward` | `avg_pool2d_backward.grad_input` | grad_out `R`, self `R`, grad_input `W` |
| `AdaptiveAvgPool2dBackward` | `_adaptive_avg_pool2d_backward.out` | grad_out `R`, self `R`, grad_input `W` |
| `Mm` | `mm.out` | a `R`, b `R`, out `W` |
| `Bmm` | `bmm.out` | a `R`, b `R`, out `W` |
| `Matmul` | `matmul.out` | a `R`, b `R`, out `W` |
| `Linear` (no bias) | `linear.out` | input `R`, weight `R`, out `W` |

Family codelet `torch_ternary`:

| `TorchKind` | Aten | Handles / modes |
|-------------|------|-----------------|
| `Addmm` (out ≠ self) | `addmm.out` | self `R`, mat1 `R`, mat2 `R`, out `W` |
| `Addmm` (out ≡ self) | `addmm.out` accumulate | self/out `RW`, mat1 `R`, mat2 `R` |
| `Linear` (with bias) | `linear.out` | input `R`, weight `R`, bias `R`, out `W` |
| `Sdpa` | `scaled_dot_product_attention` | q `R`, k `R`, v `R`, out `W` |

Specialized codelets:

| Codelet | Aten | Handles / modes |
|---------|------|-----------------|
| `torch_embedding` | `embedding.out` | weight `R`, indices `R`, out `W` |
| `torch_embedding_dense_backward` | `embedding_dense_backward.out` | grad `R`, indices `R`, grad_weight `W` |
| `torch_cat` | `cat.out` | each input `R`, out `W` |
| `torch_layer_norm` | `native_layer_norm` | input `R`; optional weight/bias `R`; out / mean / rstd `W` |
| `torch_layer_norm_backward` | `native_layer_norm_backward` | grad_out / input / mean / rstd `R`; optional weight/bias `R`; needed grad outs `W` |
| `torch_sdpa_backward` | flash-CPU bwd; CUDA: efficient (math fallback) | q / k / v / grad_out `R`; optional mask `R`; grad_q / grad_k / grad_v `W` |
| `torch_nll_loss_forward` | `nll_loss_forward.output` | log_probs `R`, target `R`, loss `W`, total_weight `W` |
| `torch_nll_loss_backward` | `nll_loss_backward.grad_input` | grad_output / log_probs / target / total_weight `R`, grad_input `W` |
| `torch_convolution` | `convolution` | input / weight `R`; optional bias `R`; out `W` |
| `torch_convolution_backward` | `convolution_backward` | grad_out / input / weight `R`; needed grad outs `W` |
| `torch_max_pool2d_with_indices` | `max_pool2d_with_indices.out` | input `R`, out `W`, indices `W` |
| `torch_max_pool2d_with_indices_backward` | `max_pool2d_with_indices_backward.grad_input` | grad_out / input / indices `R`, grad_input `W` |
| `torch_native_batch_norm` | `native_batch_norm` | input `R`; optional weight/bias `R`; running stats `RW` when training; out / saved stats `W` |
| `torch_native_batch_norm_backward` | `native_batch_norm_backward` | grad_out / input / optional stats `R`; needed grad outs `W` |

Classic I/O kept on this path (not torch-native compute, but same rules):

| Op | Modes |
|----|-------|
| `fill` | data `W` |
| `subcopy` / `copy` | src `R`, dst `W` (intersection variants may differ) |
| `clear` | data `W` |

**Scratch:** none of the current torch-native codelets register
`STARPU_SCRATCH`. ATen may allocate temporary memory inside the
kernel; that is outside StarPU’s data handles.

Example sketch (CPU; CUDA wraps with `TorchCudaEnv` — see above):

```cpp
void TorchBinary::cpu(void *buffers[], void *cl_args) noexcept
{
    auto *args = reinterpret_cast<TorchDispatchArgs *>(cl_args);
    // Prefer torch_blob::blob_fp32 (empty deleter, TLS device).
    // For TorchKind::Add: at::add_out(out, a, b, args->scalars[0]);
    // For TorchKind::Mul: at::mul_out(out, a, b);
}
```

Disable autograd / Variable mode inside the codelet
(`AutoDispatchBelowADInplaceOrView` + `NoGradGuard`) so the call hits the
CPU/CUDA aten kernel, not another PrivateUse1 recording path.

### 7. Linking

This path **links libtorch** from the StarPU/codelet side (or from a
dedicated optional target). Keep classic `nntile::kernel` CPU/CUDA codelets
torch-free. Prefer an optional CMake switch or confine torch-backed
codelets to a library that already depends on LibTorch
(`libtorch_nntile` / a sibling), so plain `libnntile` builds stay usable
without Torch where required.

Set `OMP_NUM_THREADS=1` / `torch.set_num_threads(1)` when running under
StarPU workers to avoid oversubscription.

## Match `device=cuda` registration

Baseline for every new aten schema: **how stock PyTorch dispatches it on
CUDA**, not “register PrivateUse1 and invent AutogradPrivateUse1.” Inspect
with:

```python
import torch
print(torch._C._dispatch_dump("aten::rsqrt"))
print(torch._C._dispatch_dump_table("aten::rsqrt"))
```

Compare the `CUDA:` / `AutogradCUDA:` / `PrivateUse1:` rows (load
`torch_nntile` only when checking *our* overrides).

### Dispatch layers (simplified)

```text
User call (requires_grad possible)
  → AutogradCUDA / AutogradPrivateUse1   # formula or composite
  → ADInplaceOrView                      # view / inplace bookkeeping
  → CUDA / PrivateUse1 / Composite*      # device or math kernel
```

| CUDA table row | Meaning | What to do on nntile |
|----------------|---------|----------------------|
| `RegisterCUDA` / structured `.out` delegate | Real device kernel | Register **PrivateUse1** forward (and `.out`). Leave Autograd on generic **VariableType** unless the formula’s callees are missing. |
| `CompositeImplicitAutograd` | Shared math: decomposes into other aten ops | **Do not** register PrivateUse1. Let the composite run; implement the **primitives** it lowers to (`as_strided`, `addmm`, `mm`, …). |
| `CompositeExplicitAutograd` (shared default) | Same idea; may be overridden per backend | Prefer composite. Override PrivateUse1 only if nntile storage needs a hook (e.g. `alias` must keep `TensorRef`). |
| `AutogradCUDA` = `VariableType` | Formula in `derivatives.yaml` | **Do not** add AutogradPrivateUse1. Ensure formula ops exist on nntile (`pow`, `mul`, …). |
| `AutogradCUDA` = same CompositeImplicit | Autograd *is* the decomposition | Same as composite forward — no PrivateUse1 override. |

`AutogradPrivateUse1` is **rare**. Use it only when the generic path is wrong
for nntile storage (today: `contiguous` densify under autograd). Do **not**
use it to reimplement a VariableType formula (that was the `rsqrt` mistake).

### Worked examples

**`aten::rsqrt` (pointwise device op)**

- CUDA: structured forward → `.out` device kernel; AutogradCUDA =
  VariableType; formula `-0.5 * grad * result.pow(3)`
  (`torchgen/.../derivatives.yaml`).
- nntile: PrivateUse1 `rsqrt` / `rsqrt.out` only. Implement `pow.Tensor_Scalar`
  for exponent `2`/`3` on StarPU (mul chain). Expect `grad_fn` =
  `RsqrtBackward0`, same as CPU/CUDA.
- Wrong: AutogradPrivateUse1 custom `Function` that duplicates the formula.

**`aten::linear` / `aten::matmul`**

- CUDA: both keys are CompositeImplicit (`linear` → `addmm`, `matmul` →
  `mm` / `bmm` / …). Autograd differentiates the **expanded** graph
  (`AddmmBackward0`, not a PrivateUse1 `linear_backward` registration).
- nntile: do **not** register PrivateUse1 `linear` / `matmul`. Keep
  PrivateUse1 `addmm` / `mm` / `bmm`. Helpers in `nntile_linear.cpp` /
  `nntile_gemm.cpp` may remain for StarPU/TensorGraph but must not
  `TORCH_LIBRARY_IMPL`.

**`aten::chunk` / `split` / `narrow` / `select.int`**

- CUDA: composite (or shared CompositeExplicit) → views via `narrow` /
  `as_strided` / slice. Autograd uses `SplitBackward0` / `SliceBackward0` /
  `SelectBackward0`.
- nntile: do **not** register PrivateUse1 `chunk` / `split` / `narrow` /
  `select`. Implement PrivateUse1 **`as_strided`** (and `alias`) so
  composites keep `TensorRef`.
- Wrong: PrivateUse1 `chunk` that builds views but skips Autograd’s Split
  formula (NeoX QKV backward broke that way).

### Intentional deviations

| Schema | Why not pure CUDA mirroring |
|--------|------------------------------|
| `as_strided`, `alias` | Must call `record_view_alias` / share `TensorRef`; stock composite `alias` drops BackendMeta. |
| `contiguous` (+ AutogradPrivateUse1) | Torch “contiguous” views can still be partial covers of a StarPU logical; we densify. |

### Copy-into-view (Slice / AsStrided Backward)

Composite view **forward** matches CUDA. Backward (`SliceBackward0`,
`SelectBackward0`, `AsStridedBackward0`, …) does
`zeros_like(base).as_strided(...).copy_(grad)`.

nntile→nntile `_copy_from` must **not** SSA-rebind `TensorRef` onto a
partial / strided destination (that left the base zeros untouched). For
those destinations, copy does host RMW:

1. `gather_nntile_view_to_cpu(src)` (contiguous logical of the source)
2. `gather_full_logical_to_cpu(dst)` (full parent buffer)
3. Patch `full.as_strided(dst.sizes(), dst.strides(), dst.offset()).copy_(src)`
4. `overwrite_bound_nntile_logical_from_cpu(full, dst)` (scatter into the
   existing logical; view + base keep the same `TensorRef`)

Dense full-cover nntile→nntile copies still SSA-rebind (graph style).
Do **not** re-register PrivateUse1 `narrow` / `select` to “fix” backward —
VariableType has no narrow derivative (`NotImplemented`).

Op list and policy summary:
[torch_nntile_aten_ops.md](torch_nntile_aten_ops.md).

## Checklist for a new op

1. **Dispatch audit (required):** `torch._C._dispatch_dump_table("aten::…")`
   for CUDA vs Composite vs VariableType. Follow
   [Match device=cuda registration](#match-devicecuda-registration).
   Register PrivateUse1 only for device/storage primitives; never shadow
   CompositeImplicit Autograd without a documented reason.
2. PrivateUse1 (when needed): register functional / out / inplace; meta-probe
   for non-obvious shapes; record TensorGraph op with full tensor meta.
   Prefer densifying non-contiguous inputs in the impl (`contiguous()`)
   rather than hard-failing — VariableType formulas often pass views.
3. Tensor lower: single-tile only → one TileGraph op; copy meta onto
   `TileNode`s.
4. Tile execute → `nntile::core::torch_*` / `TorchKind` (named torch-based)
   with `Tile<T>` + meta per tensor.
5. StarPU: `args_t` with all metas; CPU wrapper `from_blob` + same
   `*_out` under `NoGradGuard`; `#ifdef NNTILE_USE_CUDA` CUDA wrapper
   via `TorchCudaEnv` (StarPU stream + cuBLAS; never from the tensor);
   register `cuda_funcs`; no `nntile::kernel`.
   **Declare access modes** for every tensor (`R` / `W` / `RW` /
   `SCRATCH`) and wire them in `submit`; document the row in the
   [Data access modes](#data-access-modes) table.
6. **Autograd check:** with `requires_grad=True`, assert `type(y.grad_fn)`
   matches CPU/CUDA for the same op (e.g. `RsqrtBackward0`,
   `AddmmBackward0`). If you registered AutogradPrivateUse1, justify why
   VariableType / composite is insufficient.
7. **C++ tests (required)** — same layering as classic libnntile, under
   `nntile/tests/torch_native/` (label `torch_native`):
   - **starpu:** codelet submit vs CPU `aten::*_out` reference
   - **core:** `core::torch_*_out` vs the same aten reference
   - **tile:** TileGraph `execute` vs aten
   - **tensor:** TensorGraph structure + untiled lower/execute vs aten  
   Do **not** add multi-tile tests until tiling is re-enabled for this path.
8. **libtorch_nntile C++:** add a Catch2 case in
   `torch_nntile/tests/aten_ops_parity.cc` (label `libtorch_nntile`) for
   the PrivateUse1 schema vs CPU (fwd, and bwd when autograd applies).
   Prefer CTest / Catch2 over pytest for these aten ops.
9. Update the op table in [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md).

Run layer + PrivateUse1 suites (with `BUILD_TESTING=ON`):

```bash
ctest --test-dir build -L torch_native --output-on-failure
ctest --test-dir build -L libtorch_nntile --output-on-failure
```

CI (`.github/workflows/build-test.yml`) mirrors this on PRs to `graph_api`:
`test-libnntile` runs `-L torch_native`, `test-libtorch-nntile` runs
`-L libtorch_nntile`, and both use `--no-tests=error` so an empty suite
cannot green-pass. Layer-1 libnntile builds with LibTorch because
`NNTILE_TORCH_NATIVE_OPS` puts aten codelets inside `libnntile`.

## Why meta probe (not hand-written shapes)

In-tree CPU/CUDA share structured **`meta`** for output shapes. Calling the
op on `device=meta` reuses that logic so PrivateUse1 does not drift when
Torch changes shape rules. Use hand-written shapes only for trivial cases
(same-shape unary, obvious broadcasts already covered by helpers).

Data-dependent shapes (`nonzero`, etc.) cannot be inferred from meta alone;
those ops need a different strategy or remain unsupported on this path.

## First implementation notes

The experimental flag **`NNTILE_TORCH_NATIVE_OPS`** (default ON on
`graph_api_torch_kernels`) strips classic compute kernels/codelets/ops from
the build and keeps I/O (`fill` / `subcopy` / `clear` / `copy` / `scatter` /
`gather` / `invalidate`) plus torch-native family codelets
(`torch_dispatch` / `TorchKind`, including `Add` on `torch_binary`).

Lessons from wiring the first ops:

1. **LibTorch must link into `libnntile`** when the StarPU codelet calls ATen
   (`from_blob` + `*_out`). Confining torch solely to `libtorch_nntile`
   is not enough if torch-native codelets live in libnntile.
2. **Codelet tensors must be `device=CPU` or `device=CUDA`**, never
   `nntile`, or the call re-enters PrivateUse1. Use
   `AutoDispatchBelowADInplaceOrView` + `NoGradGuard` around `*_out`.
   On CUDA, construct blobs only after `TorchCudaEnv` (StarPU stream +
   cuBLAS); do not read stream/handle from the tensor.
3. **`from_blob` needs an empty deleter** so temporary `Tensor` destructors
   do not free StarPU memory. Prefer `torch_blob::blob_*` helpers.
4. **Tile meta:** packed `TorchTileMeta` carries sizes, strides, and
   `storage_offset` (elements) so untiled views (`transpose` / `narrow` /
   `split`) share the parent StarPU buffer. Empty packed ndim falls back
   to `core::make_contiguous_torch_meta(tile_shape)`.
5. **Functional `add.Tensor` still needs a PrivateUse1 impl** (meta probe →
   `empty` → record `torch_binary(Add, …)`). Registering only `add.out` is
   not enough. Same for `add_.Tensor` (SSA rebind via `register_data_node`).
6. Map Torch `add(self, other, alpha)` to `TorchKind::Add` with
   `extra.scalars[0] = alpha` (`out = self + alpha * other`).
7. Keep **`OMP_NUM_THREADS=1`** / `torch.set_num_threads(1)` under StarPU
   workers.
8. **Slim the pybind `_C` module** when ops are stripped from
   `libtorch_nntile`. Full `nntile_module.cpp` still binds gemm / sum_slice /
   models / etc.; under `NNTILE_TORCH_NATIVE_OPS`, build
   `nntile_module_torch_native.cpp` instead (`setup.py` picks it from
   `defs.h`). Otherwise `_C.so` fails to import with undefined symbols.
9. Gate Python package side imports (`loss`, `nn`, …) on
   `TORCH_NATIVE_OPS` in `_build_info.py` so the slim wheel stays importable.
10. Prefer **C++ CTest** for aten parity (`aten_ops_parity.cc`,
    `nntile/tests/torch_native/`). Pytest is not the coverage path for
    these torch-native ops.
11. **Match CUDA dispatch before registering.** PrivateUse1
    `chunk`/`split`/`linear`/`matmul` and AutogradPrivateUse1 `rsqrt`
    looked convenient but diverged from CUDA Autograd (Split / Addmm /
    RsqrtBackward). Audit with `_dispatch_dump_table`; see
    [Match device=cuda registration](#match-devicecuda-registration).
12. **Densify in PrivateUse1 compute impls** when VariableType formulas
    pass non-contiguous views (`mul.Scalar`, `sum`, …). Hard
    `is_contiguous` checks fail CUDA-identical backward graphs.
13. **nntile→nntile copy into views** uses host RMW +
    `overwrite_bound_nntile_logical_from_cpu` (not SSA rebind), so Slice /
    AsStrided Backward match CUDA.

Reference sources:
`nntile/src/{starpu,core,tile/ops,tensor/ops}/torch_dispatch.*`,
`nntile/include/nntile/starpu/torch_{blob,cuda_env}.hh`,
`nntile/tests/torch_native/`,
`torch_nntile/csrc/nntile_add.cpp`,
`torch_nntile/csrc/nntile_trig.cpp`,
`torch_nntile/csrc/nntile_kernels.cpp`,
`torch_nntile/csrc/nntile_executor_torch_native.cpp`,
`torch_nntile/csrc/nntile_module_torch_native.cpp`,
`torch_nntile/tests/aten_ops_parity.cc`,
`torch_nntile/tests/smoke_add.cc`.
