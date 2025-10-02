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
void cuda_kernel(Index num_elements, Scalar alpha, Scalar beta, T* dst)
{
    using Y = typename T::repr_t;
    const Y alpha_val{alpha}, beta_val{beta};
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_elements)
    {
        dst[i] = T{alpha_val + beta_val * Y{dst[i]}};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index num_elements, Scalar alpha, Scalar beta,
        T *dst)
    noexcept
//! Add scalar to buffer on CUDA
/*! Perform element-wise operation: dst[i] = alpha + beta * dst[i]
 *
 * This operation modifies the destination buffer in-place by adding a scalar
 * value (alpha) and scaling the existing values by a scalar factor (beta).
 * The operation is performed in parallel on the CUDA device.
 *
 * @param[in] stream: CUDA stream for asynchronous execution
 * @param[in] num_elements: Number of elements in the destination buffer
 * @param[in] alpha: Scalar value to add to each element
 * @param[in] beta: Scalar multiplier for each element before adding alpha
 * @param[inout] dst: Destination buffer to modify in-place
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
void cuda<fp16_t>(cudaStream_t stream, Index num_elements, Scalar alpha,
        Scalar beta, fp16_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index num_elements, Scalar alpha,
        Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_scalar
