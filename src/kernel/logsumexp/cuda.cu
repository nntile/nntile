/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/logsumexp/cuda.cu
 * Logsumexp after computed maxsumexp result of a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-30
 * */

#include "nntile/kernel/logsumexp/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace logsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T * __restrict__ maxsumexp,
        T * __restrict__ logsumexp)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        logsumexp[i] = maxsumexp[2*i] + ::log(maxsumexp[2*i+1]);
    }
}

template<>
__global__
void cuda_kernel<fp32_t>(Index nelems, const fp32_t * __restrict__ maxsumexp,
        fp32_t * __restrict__ logsumexp)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    float2 *ptr = (float2 *)maxsumexp;
    if(i < nelems)
    {
        logsumexp[i] = ptr[i].x + (ptr[i].y);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *maxsumexp, T *logsumexp)
    noexcept
//! Logsumexp of buffer
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] maxsumexp: Input buffer, result of maxsumexp operation 
 * @param[out] logsumexp: Output buffers that contains output in the end
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, maxsumexp,
            logsumexp);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *maxsumexp,
        fp32_t *logsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *maxsumexp,
        fp64_t *logsumexp)
    noexcept;

} // namespace logsumexp
} // namespace kernel
} // namespace nntile

