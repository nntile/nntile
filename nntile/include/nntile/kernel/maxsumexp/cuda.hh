/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maxsumexp/cuda.hh
 * Max and sum of exponents of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::maxsumexp
{

//! Max and sum of exponents along middle axis
/*! For a provided m-by-k-by-n input array src compute maximums and sums of
 * exponents of slices along second axis with k elements, resulting in
 * 2-by-m-by-n output array maxsumexp.
 *
 *      old[0,i,j] = maxsumexp[0,i,j]
 *      old[1,i,j] = maxsumexp[1,i,j]
 *      maxsumexp[0,i,j] = max(old[0,i,j], max(src[i,:,j]))
 *      maxsumexp[1,i,j] = old[1,i,j]*exp(old[0,i,j]-maxsumexp[0,i,j])
 *          + sum(exp(src[i,:,j]-maxsumexp[0,i,j])))
 *
 * With beta=0, old is ignored and maxsumexp is overwritten (STARPU_W).
 * With beta=1, results accumulate into maxsumexp (STARPU_RW).
 *
 * @param[in] m: Size of the first mode of src and the second mode of maxsumexp
 * arrays.
 * @param[in] n: Size of the last mode of src and maxsumexp arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: 0.0 overwrite, 1.0 accumulate
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array
 */
template <typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
          Scalar beta, T *maxsumexp) noexcept;

//! Chunk-aware maxsumexp on CUDA (ONE launch)
/*! Same operation and arithmetic order as cuda() above, but the src and
 * maxsumexp flat ranges are backed by chains of equal-size chunks. Each
 * operand's chunk base-pointer table is built by the driver through the
 * device chunk-ptr ring and travels as `*_chunks`. Every fiber
 * element resolves its chunk from its flat offset (chunk = offset / elems,
 * in-chunk offset = offset % elems; shift/mask on a power-of-two elems).
 * The per-thread reduction partition, the max/atomicAdd combine and the
 * m==1 tree are unchanged, so the result is bitwise-equal to cuda()'s up
 * to the contiguous kernel's own shared-atomic ordering floor.  Piece 7:
 * an m==1 panel that straddles a chunk boundary takes the per-element
 * chunk-resolved read path INSIDE the one launch (no host-side straddle
 * panel list, no second kernel).
 * */
template <typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst,
        long total_src, long total_dst) noexcept;

} // namespace nntile::kernel::maxsumexp
