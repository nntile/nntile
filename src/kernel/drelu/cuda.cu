/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/drelu/cuda.cu
 * Derivative of ReLU operation on a buffer on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-01-31
 * */

#include "nntile/kernel/drelu/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace drelu
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    constexpr T one = 1.0, zero = 0.0;
    for(Index i = start; i < nelems; i += step)
    {
        T &z = data[i];
        if(z > zero)
        {
            z = one;
        }
        else
        {
            z = zero;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Inplace derivative of ReLU operation performed on CUDA
/*! @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply derivative of ReLU
 * */
{
    dim3 blocks(256), threads(32);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

} // namespace drelu
} // namespace kernel
} // namespace nntile

