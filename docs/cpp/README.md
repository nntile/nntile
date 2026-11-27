# C++ implementation overview

NNTile 2.0 has one backend, NNHaul, and one kernel layout, chunk-addressable.

## Layers

### kernel

**Namespace:** `nntile::kernel`

Raw CPU/CUDA kernels. CUDA kernels have contiguous and chunked entry points:
62 of 69 kernel families have a dedicated one-launch chunked entry; the
remaining 7 are disabled families. Chunk-aware drivers remain as a
transitional/CPU-reference path (see `docs/chunked-tiles.md`).

CPU kernels are **contiguous-only** (`void cpu(...)` taking plain pointers).
They are kept in the source tree as reference implementations for the kernel
A/B tests. NNHaul creates no CPU worker today, so no compute codelet runs on
the host from the product runtime; the only host-side codelet is the scalar
logger (`log_scalar`), which forwards one value to the logger thread. The
reference kernels stay because a future NNHaul release is expected to enable
CPU workers, and chunked CPU kernels are planned for that future enablement;
none exist yet.

### nnhaul

**Namespace:** `nntile::nnhaul`

NNHaul codelets wrapping the chunked kernel entry points, plus the chunk
pointer transport. Every tile is chunk-backed. `libnntile` links the shared
`libnnhaul` runtime.

### core / tile / tensor / runtime

`core` submits one codelet for a tile; `tile` lowers tensor operations to
tile operations; `tensor` records the autograd-facing graph; `runtime`
executes compiled tile graphs.

There is no separate codelet runtime layer in NNTile 2.0.

## Libraries

- `libnnhaul.so` — runtime engine
- `libnntile.so` — kernel → NNHaul → core → tile → tensor → runtime
- `libtorch_nntile.so` — PrivateUse1 bridge and classic models

See [../backends.md](../backends.md) and [../graph.md](../graph.md).
