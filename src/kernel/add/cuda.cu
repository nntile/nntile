/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add/cuda.cu
 * Add operation on buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Aleksandr Katrutsa
 * @date 2023-07-01
 * */

#include "nntile/kernel/add/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace add
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T alpha, const T* src, T beta, T* dst)
//! Add two buffers on CUDA
/*! dst[i] = alpha*src[i] + beta*dst[i], where alpha and beta are scalars
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = alpha*src[i] + beta*dst[i];
    }
}

template<typename T>
static __global__
void cuda_kernel_beta0(Index nelems, T alpha, const T* src, T* dst)
//! Add two buffers on CUDA
/*! dst[i] = alpha*src[i], where alpha and beta are scalars
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < nelems)
    {
        dst[i] = alpha * src[i];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, const T *src, T beta,
        T *dst)
    noexcept
//! Add two buffers on CUDA
/*! dst[i] = alpha*src[i] + beta*dst[i], where alpha and beta are scalars
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    if(beta == 0.0)
    {
        (cuda_kernel_beta0<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha,
                src, dst);
    }
    else
    {
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src,
                beta, dst);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t alpha,
        const fp32_t *src, fp32_t beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t alpha,
        const fp64_t *src, fp64_t beta, fp64_t *dst)
    noexcept;

} // namespace add
} // namespace kernel
} // namespace nntile

