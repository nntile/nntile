/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/gemm/cuda.hh
 * GEMM workspace staging (user directive, GEMM POLICY 2026-10-19): the
 * dedicated device workspace whole-multi-chunk-operand staging copies
 * through, and the fused ONE-LAUNCH gather/scatter kernels.
 *
 * @version 1.2.0
 * */

#pragma once

// Compile-time definitions
#include <nntile/defs.h>

#ifdef NNTILE_USE_CUDA

// Standard libraries
#include <cstddef>

// Third-party headers
#include <cuda_runtime.h>

namespace nntile::kernel::gemm
{

//! The gemm staging workspace (see cuda.cu for the full invariant block).
//! Plain data: base is a plain cudaMalloc'd block, never freed, never
//! resized, never reallocated; every gemm task reuses it from OFFSET 0
//! (the single in-order compute stream makes each task's
//! gather -> cuBLAS -> scatter chain stream-ordered, so consecutive
//! tasks cannot overlap their workspace use).
struct Workspace
{
    unsigned char *base = nullptr; //!< device block (256-byte aligned)
    std::size_t bytes = 0; //!< capacity: 1 GiB default, env hook may shrink
    int device = -1; //!< the ONE GPU the block was allocated on
    /*! True when the block is the NNHaul engine's preallocated
     * workspace (borrowed under NNTILE_USE_NNHAUL), not a second
     * cudaMalloc: one reservation serves both. */
    bool borrowed = false;
    /*! High-water of the staged bytes this workspace actually held. */
    std::size_t peak_bytes = 0;
};

/*! The ONE workspace of the process. Declared here, defined (NOT
 * inline) in src/kernel/gemm/cuda.cu: the function-local static's
 * ODR-guaranteed uniqueness gives exactly ONE instance in the whole
 * binary no matter how many translation units call this -- a
 * header-inline accessor could not promise that. First call allocates
 * (NNHAUL_GEMM_WORKSPACE_BYTES test hook, read once per process like
 * the ring's hooks; default exactly 1 GiB); a cudaMalloc failure is a
 * LOUD error (stderr + abort -- there is no fallback); a later call on
 * a different CUDA device throws (the workspace is per-GPU; this is a
 * single-GPU campaign). Never throws for any other reason. */
Workspace &workspace();

//! One staged operand of the fused gather/scatter: the WHOLE flat byte
//! space of a multi-chunk buffer (equal 16 MiB-class chunks plus a
//! shorter tail), its chunk-pointer table (a device chunk-ptr-ring
//! view) and its contiguous slot in the gemm workspace.
/*! (user directive, 2026-10-19: reuse the scalable ring-table kernel)
 * The staging IS kernel::copy::chunked (see src/nnhaul/gemm.cc): each
 * staged operand is a chunked->workspace (or workspace->chunked) copy
 * whose workspace side rides the ring as a 1-entry table. The
 * gemm-local fused stage kernel, StageSeg/StagePlan and their entry
 * points are deleted. This module keeps ONLY the workspace. */

} // namespace nntile::kernel::gemm

#endif // NNTILE_USE_CUDA
