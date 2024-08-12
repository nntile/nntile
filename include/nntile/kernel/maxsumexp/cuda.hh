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

#include <cuda_runtime.h>
#include <nntile/base_types.hh>

namespace nntile::kernel::maxsumexp {

//! Launch reference implementation of `maxsumexp` kernel.
//
//  This implementation is based on parallel sequential aggregation of maximum
//  element of input and rolling evalualution of exponents and their sums.
template <typename T>
void LaunchMaxSumExp1(cudaStream_t stream, Index m, Index n, Index k,
                      T const *src, T *dst) noexcept;

extern template void LaunchMaxSumExp1<fp32_t>(cudaStream_t stream, Index m,
                                              Index n, Index k,
                                              const fp32_t *src,
                                              fp32_t *maxsumexp) noexcept;

extern template void LaunchMaxSumExp1<fp64_t>(cudaStream_t stream, Index m,
                                              Index n, Index k,
                                              const fp64_t *src,
                                              fp64_t *maxsumexp) noexcept;

//! Launch accelerated implementation of `maxsumexp` kernel.
//
//  Speed up was archived through use of shared memory with block and warp
//  max/sum-reductions. This variants seems a little bit more numerically
//  stable since it does not compuate rolling sum scaled by factor \f( \exp{x -
//  x_{max}} \f)
template <typename T>
void LaunchMaxSumExp3(cudaStream_t stream, Index m, Index n, Index k,
                      T const *src, T *dst) noexcept;

extern template void LaunchMaxSumExp3<fp32_t>(cudaStream_t stream, Index m,
                                              Index n, Index k,
                                              const fp32_t *src,
                                              fp32_t *maxsumexp) noexcept;

extern template void LaunchMaxSumExp3<fp64_t>(cudaStream_t stream, Index m,
                                              Index n, Index k,
                                              const fp64_t *src,
                                              fp64_t *maxsumexp) noexcept;

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
 * @param[in] m: Size of the first mode of src and the second mode of maxsumexp
 * arrays.
 * @param[in] n: Size of the last mode of src and maxsumexp arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[inout] maxsumexp: Output contiguous 2-by-m-by-n array, that
 * accumulates sums and norms of slices along middle axis.
 */
template <typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
          T *maxsumexp) noexcept;

} // namespace nntile::kernel::maxsumexp
