/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/copy/cuda.hh
 * Copy operation on buffers on CUDA
 *
 * @version 1.2.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::copy
{

//! Byte-copy chunked buffers on CUDA (ONE launch, chunk tables via UVA)
/*! Semantically a memcpy of `nbytes` bytes; each source/destination byte is
 * resolved through its operand's device-ring chunk table. */
void chunked(cudaStream_t stream, Index nbytes,
        const unsigned char* const* src_chunks, Index nsrc,
        unsigned char* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst)
    noexcept;

//! One of up to three discontiguous chunked byte ranges.
struct Segment
{
    const unsigned char *const *src = nullptr; //!< device-ring src table
    Index nsrc = 0;
    unsigned char *const *dst = nullptr; //!< device-ring dst table
    Index ndst = 0;
    Index nbytes = 0;
    Index elems_src = 0; //!< bytes in a full src chunk
    Index elems_dst = 0; //!< bytes in a full dst chunk
};

//! Copy up to three discontiguous chunked byte ranges in ONE launch.
/*! Equivalent to one chunked() call per segment, fused into a single
 * kernel launch (grid.y selects the segment). The gemm staging path
 * uses it to gather A, B and (when beta != 0) C together. */
void chunked_segments(cudaStream_t stream, Segment const *segments,
        int nsegments)
    noexcept;

} // namespace nntile::kernel::copy
