/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod/cuda.cu
 * Per-element product of two buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-26
 * */

#include "nntile/kernel/prod/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace prod
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    for(Index i = start; i < nelems; i += step)
    {
        dst[i] *= src[i];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! Per-element product of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] src: Input buffer
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    dim3 blocks(256), threads(32);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

} // namespace prod
} // namespace kernel
} // namespace nntile

