/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/accumulate_maxsumexp/cuda.cu
 * Accumulate maxsumexp buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-09-20
 * */

#include "nntile/kernel/accumulate_maxsumexp/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace accumulate_maxsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T* src, T* dst)
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
    if(i < nelems)
    {
        if(dst[2*i] < src[2*i])
        {
            dst[2*i+1] = src[2*i+1] + dst[2*i+1]*::exp(dst[2*i]-src[2*i]);
            dst[2*i] = src[2*i];
        }
        else
        {
            dst[2*i+1] += src[2*i+1]*::exp(src[2*i]-dst[2*i]);
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

} // namespace accumulate_maxsumexp
} // namespace kernel
} // namespace nntile

