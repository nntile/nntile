/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_inplace/cuda.cu
 * Inplace SiLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::silu_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y one = Y{1.0};
    Y data_val{0.0};
    if(i < nelems)
    {
        data_val = Y{data[i]};
        data[i] = T{data_val / (one + ::exp(-data_val))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Inplace SiLU operation on CUDA
/*! Does the following per-element operation:
 * data[i] = data[i] * sigmoid(data[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply SiLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, fp16_t *data)
    noexcept;

} // namespace nntile::kernel::silu_inplace