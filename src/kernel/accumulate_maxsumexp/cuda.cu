/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_maxsumexp/cuda.cu
 * Accumulate maxsumexp buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/accumulate_maxsumexp/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::accumulate_maxsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
//! Accumulate two maxsumexp buffers on CUDA
/*! Performs the following operation:
 *      dst[2*i+1] = dst[2*i+1]*exp(dst[2*i]) + src[2*i+1]*exp(src[2*i]),
 *      dst[2*i] = max(src[2*i], dst[2*i]).
 *
 * @param[in] nelems: Number of (max,sumexp) pairs of the src and dst tensors
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination of the maxsumexp accumulation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y zero = 0.0;
    if(i < nelems)
    {
        Y dst_odd(dst[2*i+1]);
        Y dst_even(dst[2*i]);

        Y src_odd(src[2*i+1]);
        Y src_even(src[2*i]);
        // Do nothing if sum of exponents of source is zero
        if(src_odd != zero)
        {
            // Overwrite if old value of sum is zero
            if(dst_odd == zero)
            {
                dst[2*i] = src[2*i];
                dst[2*i+1] = src[2*i+1];
            }
            // Otherwise update based on maximum
            else if(dst_even < src_even)
            {
                dst[2*i+1] = src_odd + dst_odd*::exp(dst_even-src_even);
                dst[2*i] = src[2*i];
            }
            else
            {
                dst[2*i+1] = dst_odd + src_odd*::exp(src_even-dst_even);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! Accumulate two maxsumexp buffers on CUDA
/*! Performs the following operation:
 *      dst[2*i+1] = dst[2*i+1]*exp(dst[2*i]) + src[2*i+1]*exp(src[2*i]),
 *      dst[2*i] = max(src[2*i], dst[2*i]).
 *
 * @param[in] nelems: Number of (max,sumexp) pairs of the src and dst tensors
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination of the maxsumexp accumulation
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *src,
        fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *src,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *src,
        bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::accumulate_maxsumexp
