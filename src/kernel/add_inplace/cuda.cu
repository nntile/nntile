/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_inplace/cuda.cu
 * Add operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_inplace
{

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T *src, Scalar beta_,
        T *dst)
//! Generic implementation of the add_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_inplace::cuda
 * */
{
    Index i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    Y dst_block[LOOP];
    Y src_block[LOOP];
    Y alpha = Y{alpha_};
    Y beta = Y{beta_};
    constexpr Index BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(Index j = 0; j < LOOP; ++j)
        {
            dst_block[j] = static_cast<Y>(dst[i+j*BLOCK_STEP]);
            src_block[j] = static_cast<Y>(src[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*src_block[j] + beta*dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
    else
    {
        Index j_max = (nelems-i+BLOCK_STEP-1) / BLOCK_STEP;
        for(Index j = 0; j < j_max; ++j)
        {
            dst_block[j] = static_cast<Y>(dst[i+j*BLOCK_STEP]);
            Y val1 = static_cast<Y>(src[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*val1 + beta*dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha_, const T *src_,
        Scalar beta_, T *dst_)
    noexcept
//! Add two buffers inplace with optional scaling inplace on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and dst[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace instead, and in a case of beta being zero,
 * use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add operation
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems, alpha_,
            src_, beta_, dst_);
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

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_inplace
