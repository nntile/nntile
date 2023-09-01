/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot/cuda.cu
 * hypot operation on buffers on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-03
 * */

#include "nntile/kernel/hypot/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace hypot
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T alpha, const T* src, T beta, T* dst)
//! hypot two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src[i], beta*dst[i]),
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the hypot operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    constexpr T zero = 0.0;
    if(i < nelems)
    {
        if(alpha == zero)
        {
            if(beta == zero)
            {
                dst[i] = zero;
            }
            else
            {
                dst[i] = ::abs(beta * dst[i]);
            }
        }
        else
        {
            if(beta == zero)
            {
                dst[i] = ::abs(alpha * src[i]);
            }
            else
            {
                dst[i] = ::hypot(alpha*src[i], beta*dst[i]);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T alpha, const T *src, T beta,
        T *dst)
    noexcept
//! hypot two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src[i], beta*dst[i]),
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the hypot operation
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

} // namespace hypot
} // namespace kernel
} // namespace nntile

