/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/multiply_inplace/cuda.hh
 * Per-element product of two buffers on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::multiply_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src, T *dst)
    noexcept;

//! Per-element product of two chunked buffers on CUDA
/*! Same operation as cuda() above (dst[i] = alpha * src[i] * dst[i] for
 * every flat element i in [0, nelems)), but both operands' flat element
 * ranges are backed by a chain of equal-size chunk allocations (a shorter
 * tail on the last chunk). The WHOLE nelems range is issued as ONE kernel
 * launch: the two chunk base-pointer tables are host arrays in pinned,
 * device-mapped memory (built by the driver through the shared
 * ChunkPtrRing) whose device-ring views travel as `src_chunks`/`dst_chunks`,
 * and each block stages them into shared memory once. Each element resolves
 * its src chunk from its flat offset via i / elems_src, i % elems_src and its
 * dst chunk via i / elems_dst, i % elems_dst (shift/mask on the compile-time
 * chunk class). The per-element arithmetic is exactly cuda()'s, so the
 * result is BITWISE the whole-buffer cuda() call. There is no chunk-count cap.
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] alpha: Scalar multiplier
 * @param[in] src_chunks: Base pointers of src's chunks, in order
 * @param[in] nsrc: Number of src chunks
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] dst_chunks: Base pointers of dst's chunks, in order
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
    const T *const *src_chunks, Index nsrc, Index elems_src,
    T *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::multiply_inplace
