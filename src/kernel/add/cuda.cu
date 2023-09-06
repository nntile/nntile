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
 * @author Konstantin Sozykin
 * @date 2023-09-06
 * */

#include "nntile/kernel/add/cuda.hh"
#include <cuda_fp16.h>



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
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
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
void cuda(cudaStream_t stream, Index nelems, T alpha, const T *src, T beta,
        T *dst)
    noexcept
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src, beta,
            dst);
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

// Explicit instantiation of cuda function for fp16 type
template<typename T> // spefic template for fp16_t
void cuda(cudaStream_t stream, Index nelems, fp32_t alpha, const T *src, fp32_t beta,
        T *dst)
    noexcept;

template<>
void cuda<fp16_t>(cudaStream_t stream, Index nelems, fp32_t alpha, const fp16_t *src, fp32_t beta,
        fp16_t *dst)
    noexcept
//! Add two buffers on CUDA in half precission, see in destiction in original template
{

    dim3 blocks((nelems+255)/256), threads(256);
    __half alpha_half = __float2half(alpha);
    __half beta_half = __float2half(beta); 
    const __half *src_half = reinterpret_cast<const __half *>(src);
    __half *dst_half = reinterpret_cast<__half *>(dst);
    (cuda_kernel<__half>)<<<blocks, threads, 0, stream>>>(nelems, alpha_half, src_half, beta_half,
            dst_half);
}


} // namespace add
} // namespace kernel
} // namespace nntile

