/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/norm_slice_inplace/cuda.hh
 * Euclidean norms of fibers into a slice of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::norm_slice_inplace
{

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept;

//! Euclidean norms of chunked fibers into a slice on CUDA
/*! Same operation as cuda() above, but the source's and destination's flat
 * element ranges are backed by a chain of equal-size chunk allocations (a
 * shorter tail on the last chunk). The whole in-place update is issued as
 * ONE kernel launch: the two chunk base-pointer tables are host arrays in
 * device-resident memory (built by the driver through the shared
 * ChunkPtrRing) whose device-ring views travel as the `*_chunks` args, and
 * each block stages them into shared memory once. Every element derives its
 * chunk index from its own flat offset (chunk = offset / elems, in-chunk
 * offset = offset % elems; a shift/mask on the compile-time chunk class; a single-chunk operand uses the sentinel shift 62 so there is no runtime-divisor path). The
 * kernel reproduces, for each of cuda()'s branches, that branch's thread
 * geometry and accumulation order (the m==1 kernel's strided per-thread hypot
 * and shared tree reduction; the general kernel's blocked per-fiber hypot
 * with the shared max-rescale and atomicAdd reduction), so the result equals
 * cuda()'s up to the same atomic ordering nondeterminism the general branch
 * already has - and is bitwise cuda()'s on the m==1 branch, which has no
 * atomics. There is no chunk-count cap.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of src and dst
 * @param[in] n: Size of the last mode of src and dst
 * @param[in] k: Size of the middle mode of src
 * @param[in] alpha: Scaling factor for src
 * @param[in] src_chunks: Base pointers of src's chunks, in order
 * @param[in] nsrc: Number of src chunks
 * @param[in] beta: Scaling factor for dst
 * @param[in] dst_chunks: Base pointers of dst's chunks, in order
 * @param[in] ndst: Number of dst chunks
 * @param[in] elems_src: Elements in a full src chunk (uniform)
 * @param[in] elems_dst: Elements in a full dst chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::norm_slice_inplace
