/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu/cuda.cu
 * ReLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/relu/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::relu
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    using Y = typename T::repr_t;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    constexpr Y zero = 0;
    if(i < nelems)
    {
        data[i] = T{::fmax(Y(data[i]), zero)};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Inplace ReLU operation on CUDA
/*! Does the following per-element operation:
 * ReLU(z) = max(z, 0)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply ReLU
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

} // namespace nntile::kernel::relu
