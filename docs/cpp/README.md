<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file docs/cpp/README.md
 @version 2.0.0
-->

# C++ implementation overview

NNTile 2.0 has one backend, NNHaul, and one kernel layout, chunk-addressable.

## Layers

### kernel

**Namespace:** `nntile::kernel`

Raw CPU/CUDA kernels. CUDA kernels have contiguous and chunked entry points:
62 of 69 kernel families have a dedicated one-launch chunked entry; the
remaining 7 are disabled families. No active family uses a chunk-aware
piecewise driver; the driver mode is retained only for the disabled families
and the CPU-reference path (see `docs/chunked-tiles.md`).

CPU kernels are **contiguous-only** (`void cpu(...)` taking plain pointers).
They are kept in the source tree as reference implementations for the kernel
A/B tests. NNHaul's host lane exists, but the NNTile product submits no CPU compute
codelets, so no compute runs on the host; the only host-side codelet is the
scalar logger (`log_scalar`), which forwards one value to the logger thread. The
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
