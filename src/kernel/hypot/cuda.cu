/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot/cuda.cu
 * hypot operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/hypot/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::hypot
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T* src, Scalar beta_, T* dst)
//! hypot two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = hypot(alpha*src[i], beta*dst[i]),
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the hypot operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};
    Y alpha{alpha_};
    Y beta{beta_};
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
                dst[i] = ::fabs(beta * Y{dst[i]});
            }
        }
        else
        {
            if(beta == zero)
            {
                dst[i] = ::fabs(alpha * Y{src[i]});
            }
            else
            {
                dst[i] = ::hypot(alpha*Y{src[i]}, beta*Y{dst[i]});
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src,
        Scalar beta, T *dst)
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
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src,
            beta, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::hypot
