/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_scalar/cuda.cu
 * Add scalar operation of buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_scalar/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_scalar
{

template<typename T>
static __global__
void cuda_kernel(Index num_elements, Scalar alpha_, Scalar beta_, T* dst)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_}, beta{beta_};
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_elements)
    {
        dst[i] = T{alpha + beta*Y{dst[i]}};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_elements, Scalar alpha, Scalar beta,
        T *dst)
    noexcept
//! Add scalar to buffer buffers on CUDA
/*! dst[i] = alpha + beta*dst[i], where alpha and beta are scalars
 *
 * @param[in] num_elements: Size of the src and dst tensors
 * @param[in] alpha: Scalar bias for the dst tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add_scalar operation
 * */
{
    dim3 blocks((num_elements+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(num_elements, alpha, beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index num_elements, Scalar alpha,
        Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index num_elements, Scalar alpha,
        Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_elements, Scalar alpha,
        Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_scalar
