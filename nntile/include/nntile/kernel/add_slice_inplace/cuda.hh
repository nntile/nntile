/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_slice_inplace/cuda.hh
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::add_slice_inplace
{

// Per-element addition of a tensor and a broadcasted slice on CUDA
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept;

//! Add a chunked slice into a chunked tensor in place on CUDA (ONE launch)
/*! Same per-element operation as cuda() above (including the m == 1 special
 * case, whose arithmetic is identical per element), but src and dst flat
 * ranges are backed by chains of equal-size chunks. One kernel launch over
 * the whole (m, n, k) range; each operand's chunk base-pointer table is a
 * host array in device-resident memory (built by the driver through the
 * device chunk-ptr ring) whose device-ring view travels as the `*_chunks`
 * args, and each block stages it into shared memory once. Every element
 * resolves its chunk from its flat offset (chunk = offset / elems, in-chunk
 * offset = offset % elems; shift/mask on a power-of-two elems). The result is
 * bitwise-equal to cuda()'s. There is no chunk-count cap.
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::add_slice_inplace
