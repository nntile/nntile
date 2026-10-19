/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gemm/cuda.cu
 * GEMM workspace staging (user directive, GEMM POLICY 2026-10-19):
 *
 * 1. ONE plain cudaMalloc'd workspace, default exactly 1 GiB, allocated
 *    ONCE per process on first use and NEVER freed, resized or
 *    reallocated -- the chunk-ptr ring's discipline. Exactly ONE
 *    instance exists in the whole binary: the accessor's
 *    function-local static has C++'s ODR-guaranteed uniqueness, and the
 *    accessor is a NON-inline function with this file as its single
 *    definition site (a header-inline accessor could produce one
 *    instance per translation unit's inline slot -- the ODR would
 *    still nominally merge them, but the single-definition form leaves
 *    no room for that surprise at all).
 * 2. The workspace is PER GPU: it is bound to the CUDA device current
 *    at first use; a later call on a different device throws (this is a
 *    single-GPU campaign; the guard keeps the binding honest if that
 *    ever changes).
 * 3. The workspace is touched ONLY by the COMPUTE stream: the fused
 *    gather kernel, the cuBLAS call and the fused scatter kernel of a
 *    gemm task all run on ::nnhaul::cuda_stream(). Nothing else may
 *    ever read or write workspace memory -- not the chunk transport's
 *    control stream (it carries only pointer-table bytes), not the data
 *    manager's offload/prefetch copy streams, not any memset on
 *    another stream. The offset-0 stream-ordered reuse design DEPENDS
 *    on this invariant: each task's gather -> cuBLAS -> scatter chain
 *    is ordered on the one compute stream, so consecutive tasks cannot
 *    overlap their workspace use and every task reuses the workspace
 *    from OFFSET 0 (no allocator, no free list).
 *
 * A cudaMalloc failure is a LOUD error (stderr + abort): there is no
 * fallback and no smaller workspace (rule 11). The overflow check
 * (staged bytes against the capacity) is POLICY and lives in the gemm
 * wrapper (src/nnhaul/gemm.cc), which knows the operand names.
 *
 * @version 1.2.0
 * */

// Corresponding headers
#include "nntile/kernel/gemm/cuda.hh"

// Standard libraries
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

// Other NNTile headers
#include "nntile/base_types.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gemm
{

//! Test hook: NNHAUL_GEMM_WORKSPACE_BYTES shrinks the workspace so the
//! wrapper's overflow check can be exercised cheaply. Read ONCE per
//! process (the ring hooks' discipline). Default: exactly 1 GiB.
static std::size_t workspace_bytes_from_env() noexcept
{
    char const *e = std::getenv("NNHAUL_GEMM_WORKSPACE_BYTES");
    if(e != nullptr)
    {
        long const v = std::atol(e);
        if(v > 0)
        {
            return static_cast<std::size_t>(v);
        }
        std::fprintf(stderr,
            "nntile gemm: NNHAUL_GEMM_WORKSPACE_BYTES=%s is not a positive "
            "integer; keeping the default 1 GiB\n",
            e);
    }
    return std::size_t{1} << 30;
}

//! The ONE workspace (see the file header for the full invariants).
//! Non-inline, single definition site: this exact function object's
//! function-local static is the process's only instance.
Workspace &workspace()
{
    static Workspace w = []() {
        Workspace s;
        s.bytes = workspace_bytes_from_env();
        int dev = -1;
        cudaError_t const gd = cudaGetDevice(&dev);
        if(gd != cudaSuccess)
        {
            std::fprintf(stderr,
                "nntile gemm: workspace cudaGetDevice failed: %s\n",
                cudaGetErrorString(gd));
            std::abort();
        }
        cudaError_t const err =
            cudaMalloc(reinterpret_cast<void **>(&s.base), s.bytes);
        if(err != cudaSuccess)
        {
            std::fprintf(stderr,
                "nntile gemm: workspace cudaMalloc(%zu bytes) failed: %s -- "
                "the chunked gemm staging cannot exist (no fallback)\n",
                s.bytes, cudaGetErrorString(err));
            std::abort();
        }
        s.device = dev;
        return s;
    }();
    int dev = -1;
    if(cudaGetDevice(&dev) == cudaSuccess && dev != w.device)
    {
        throw std::runtime_error(
            "nntile gemm: the gemm workspace is bound to device "
            + std::to_string(w.device) + "; device " + std::to_string(dev)
            + " requested (the workspace is per-GPU)");
    }
    return w;
}

//! (user directive, 2026-10-19) The fused stage kernel and its
//! StagePlan machinery are deleted -- the staging is
//! kernel::copy::chunked (see src/nnhaul/gemm.cc); this module keeps
//! only the workspace.
} // namespace nntile::kernel::gemm
