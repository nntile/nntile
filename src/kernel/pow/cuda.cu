/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/pow/cuda.cu
 * Power operation on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/pow/cuda.hh"

namespace nntile::kernel::pow
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T alpha, T exp, T *data)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    for(Index i = start; i < nelems; i += step)
    {
        T z = data[i];
        data[i] = alpha * ::pow(z, exp);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, T exp, T *data)
    noexcept
//! Inplace power operation on CUDA
/*! Does the following per-element operation:
 * pow(z) = alpha * z^exp
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply power function
 * */
{
    dim3 blocks(256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, exp, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t alpha, fp32_t exp,
        fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t alpha, fp64_t exp,
        fp64_t *data)
    noexcept;

} // namespace nntile::kernel::pow
