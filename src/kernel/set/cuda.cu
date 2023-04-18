/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/set/cuda.cu
 * Set operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-18
 * */

#include "nntile/kernel/set/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace set
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T val, T *data)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    constexpr T zero = 0;
    for(Index i = start; i < nelems; i += step)
    {
        data[i] = val;
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T val, T *data)
    noexcept
//! Set operation on CUDA
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val: Input value
 * @params[out] data: Output buffer
 * */
{
    dim3 blocks(256), threads(32);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, val, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t val, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t val, fp64_t *data)
    noexcept;

} // namespace set
} // namespace kernel
} // namespace nntile

