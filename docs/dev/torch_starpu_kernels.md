# Adding a non-tiled torch kernel as a StarPU codelet

**Status:** design for `graph_api_torch_kernels`  
**Scope:** single-tile (untiled) tensors on `device=nntile` only  
**Related:** [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md),
[../cpp/README.md](../cpp/README.md), [../graph.md](../graph.md)

This note describes how to plug a **libtorch / ATen** CPU kernel into the
Graph API stack so that an untiled `device=nntile` op still runs under
StarPU’s task runtime, without a matching `nntile::kernel` implementation.

CUDA wrappers are out of scope for the first cut (CPU StarPU workers only).

## Goal

```text
torch / autograd
    → PrivateUse1 aten impl (device=nntile)
    → TensorGraph OpNode  (record only)
    → lower (single tile only) → TileGraph OpNode
    → Runtime execute → nntile::core (torch-based)
    → nntile::starpu::submit → StarPU codelet
    → codelet CPU wrapper: from_blob + aten::*_out
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

Register functional, inplace, and out overloads separately on PrivateUse1
(`add.Tensor`, `add.out`, `add_.Tensor`, …). Out/inplace alone do **not**
make functional `add` work on PrivateUse1.

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
is clearly named as torch-based (e.g. `core::torch_add_out`), **not** a
`nntile::kernel::*` function.

### 5. `nntile::core` (torch-based, no `nntile::kernel`)

Signature sketch:

```cpp
// Torch-based core path: Tile + ATen meta → StarPU submit.
// No nntile::kernel counterpart.
void torch_add_out(
    const Tile<fp32_t> &self,
    const TileMeta &self_meta,
    const Tile<fp32_t> &other,
    const TileMeta &other_meta,
    const Tile<fp32_t> &out,
    const TileMeta &out_meta,
    Scalar alpha);
```

`TileMeta` holds strides / dtype / sizes needed to rebuild `at::Tensor`
views. Core only packs handles and calls `nntile::starpu::<op>.submit(...)`.

### 6. `nntile::starpu` codelet

Follow the existing codelet pattern (`Codelet` + `args_t` + `submit` +
CPU wrapper), with these differences:

1. **`submit`** compresses **all** input/output tensor meta into a single
   `args_t` (type unique per codelet) and passes it to
   `starpu_task_insert` / `nntile_starpu_task_insert` together with StarPU
   data handles for the tile buffers.
2. **CPU function** (`void (*)(void *buffers[], void *cl_args)`):
   - Unpack `args_t`.
   - For each buffer, build `at::Tensor` with `torch::from_blob` (or
     `at::from_blob`) using the raw pointer and the meta from `args_t`.
   - Use an **empty deleter** so destroying the temporary `Tensor` does
     **not** free StarPU-owned memory.
   - Call the low-level aten API, preferably `at::<op>_out(...)` /
     `at::_ops::<op>_out::call(...)`.
3. No CUDA implementation in the first version (`where` = `STARPU_CPU`).

Example sketch:

```cpp
void TorchAddOut::cpu(void *buffers[], void *cl_args) noexcept
{
    auto *args = reinterpret_cast<args_t *>(cl_args);
    auto **ifaces = reinterpret_cast<VariableInterface **>(buffers);
    float *self_ptr = ifaces[0]->get_ptr<float>();
    float *other_ptr = ifaces[1]->get_ptr<float>();
    float *out_ptr = ifaces[2]->get_ptr<float>();

    auto opts = at::TensorOptions()
        .dtype(at::kFloat)
        .device(at::kCPU);
    at::Tensor self = at::from_blob(
        self_ptr, args->self_sizes, args->self_strides, opts);
    at::Tensor other = at::from_blob(
        other_ptr, args->other_sizes, args->other_strides, opts);
    at::Tensor out = at::from_blob(
        out_ptr, args->out_sizes, args->out_strides, opts);

    at::add_out(out, self, other, args->alpha);
    // self/other/out destructors must not free self_ptr/other_ptr/out_ptr
}
```

Disable autograd / Variable mode inside the codelet if needed
(`AutoDispatchBelowAutograd` or equivalent) so the call hits the CPU
kernel, not another PrivateUse1 recording path.

### 7. Linking

This path **links libtorch** from the StarPU/codelet side (or from a
dedicated optional target). Keep classic `nntile::kernel` CPU/CUDA codelets
torch-free. Prefer an optional CMake switch or confine torch-backed
codelets to a library that already depends on LibTorch
(`libtorch_nntile` / a sibling), so plain `libnntile` builds stay usable
without Torch where required.

Set `OMP_NUM_THREADS=1` / `torch.set_num_threads(1)` when running under
StarPU workers to avoid oversubscription.

## Checklist for a new op

1. PrivateUse1: register functional / out / inplace as needed; meta-probe
   for non-obvious shapes; record TensorGraph op with full tensor meta.
2. Tensor lower: single-tile only → one TileGraph op; copy meta onto
   `TileNode`s.
3. Tile execute → `nntile::core::torch_<op>` (named torch-based) with
   `Tile<T>` + meta per tensor.
4. StarPU: `args_t` with all metas; CPU wrapper `from_blob` + `*_out`;
   no `nntile::kernel`.
5. Untiled parity test against CPU Torch; do **not** add multi-tile tests
   until tiling is re-enabled for this path.
6. Update the op table in [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md).

## Why meta probe (not hand-written shapes)

In-tree CPU/CUDA share structured **`meta`** for output shapes. Calling the
op on `device=meta` reuses that logic so PrivateUse1 does not drift when
Torch changes shape rules. Use hand-written shapes only for trivial cases
(same-shape unary, obvious broadcasts already covered by helpers).

Data-dependent shapes (`nonzero`, etc.) cannot be inferred from meta alone;
those ops need a different strategy or remain unsupported on this path.
