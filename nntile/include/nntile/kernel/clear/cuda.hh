/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/clear/cuda.hh
 * Clear operation on buffers on CUDA
 *
 * @version 1.2.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::clear
{

//! Zero-fill chunked buffers on CUDA (ONE launch)
/*! Semantically a memset of `nbytes` bytes to zero, rounded up to a
 * 16-byte multiple (the overclear lands in allocation padding); every
 * 16-byte unit resolves through its device-ring chunk table and stores
 * one unconditional uint4. A single kernel because per-chunk memsets do
 * not scale with bandwidth (a 16 MiB chunk is ~4 us of fill at 3.6
 * TB/s, where launch overhead dominates). */
void chunked(cudaStream_t stream, Index nbytes,
        unsigned char* const* data_chunks, Index ndata, Index elems_data)
    noexcept;

} // namespace nntile::kernel::clear
