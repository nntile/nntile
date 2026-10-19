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

} // namespace nntile::kernel::copy
