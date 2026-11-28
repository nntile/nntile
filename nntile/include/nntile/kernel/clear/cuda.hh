/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/clear/cuda.hh
 * Clear operation on buffers on CUDA
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>
#include <nntile/kernel/cuda.hh>

namespace nntile::kernel::clear
{

//! Zero-fill chunked buffers on CUDA (ONE launch)
/*! Semantically a memset of exactly `nbytes` bytes to zero. Every whole
 * 16-byte unit resolves through its device chunk table and stores one
 * unconditional uint4; a final sub-16-byte tail is stored byte-wise, so
 * no byte outside [0, nbytes) is written. A single kernel because per-chunk memsets do
 * not scale with bandwidth: at class-sized chunks the fills are short
 * enough that launch overhead dominates. */
void chunked(cudaStream_t stream, Index nbytes,
        unsigned char* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls = chunk_class())
    noexcept;

} // namespace nntile::kernel::clear
