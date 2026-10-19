/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_inplace/cuda.hh
 * Add operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::add_inplace
{

// Apply add for buffers on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src,
        Scalar beta, T *dst)
    noexcept;

//! Add two chunked buffers in place on CUDA (ONE launch, chunk tables via UVA)
/*! Same per-element operation as cuda() above, but src and dst flat ranges
 * are backed by chains of equal-size chunks. One kernel launch over the whole
 * logical range; each operand's chunk base-pointer table is a host array in
 * device-resident memory (built by the driver through the shared
 * ChunkPtrRing) whose device-ring view travels as `src_chunks`/`dst_chunks`,
 * and each block stages it into shared memory once. Every element resolves
 * its chunk from its flat offset (chunk = offset / elems, in-chunk offset =
 * offset % elems; shift/mask on a power-of-two elems). The result is
 * bitwise-equal to cuda()'s (independent elements, same expression). There is
 * no chunk-count cap.
 * */
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::add_inplace
