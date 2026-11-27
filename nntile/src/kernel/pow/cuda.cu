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
 * @version 1.1.0
 * */

#include "nntile/kernel/pow/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::pow
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, Scalar exp_, T *data)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, exp{exp_};
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    for(Index i = start; i < nelems; i += step)
    {
        Y z = Y{data[i]};
        data[i] = T{alpha * ::pow(z, exp)};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, Scalar exp,
        T *data)
    noexcept
//! Inplace power operation on CUDA
/*! Does the following per-element operation:
 * pow(z) = alpha * z^exp
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply power function
 * */
{
    dim3 blocks(256), threads(256);
    using Y = typename T::repr_t;
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, exp, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, Scalar exp,
        fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, Scalar exp,
        fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, Scalar exp,
        bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, Scalar exp,
        fp16_t *data)
    noexcept;

} // namespace nntile::kernel::pow
