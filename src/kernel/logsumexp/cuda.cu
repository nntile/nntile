/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/logsumexp/cuda.cu
 * Logsumexp operation of buffer on CUDA
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
void cuda_kernel(Index m, const T *maxsumexp, T *logsumexp)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < m)
    {
        logsumexp[i] = maxsumexp[2*i] + std::log(maxsumexp[2*i+1]);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, const T *maxsumexp, T *logsumexp)
    noexcept
//! Logsumexp of buffer
/*! One of the buffers serves as output
 *
 * @param[in] m: Number of elements in both buffers
 * @param[in] maxsumexp: Input buffer, result of maxsumexp operation 
 * @param[out] logsumexp: Output buffers that contains output in the end
 * */
{
    dim3 blocks((m+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, maxsumexp, logsumexp);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, const fp32_t *maxsumexp, fp32_t *logsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, const fp64_t *maxsumexp, fp64_t *logsumexp)
    noexcept;

} // namespace logsumexp
} // namespace kernel
} // namespace nntile

