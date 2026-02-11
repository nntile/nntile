/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add/cuda.cu
 * Add operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add
{

template<typename T, int BLOCK, Index LOOP>
static __global__
void cuda_kernel(
    Index nelems,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
)
//! Generic implementation of the add operation on CUDA
/* @copydoc nntile::kernel::add::cuda
 * */
{
    Index i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    Y dst_block[LOOP];
    Y src_block[LOOP];
    const Y alpha_ = alpha;
    const Y beta_ = beta;
    constexpr Index BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(Index j = 0; j < LOOP; ++j)
        {
            src_block[j] = static_cast<Y>(src1[i+j*BLOCK_STEP]);
            dst_block[j] = static_cast<Y>(src2[i+j*BLOCK_STEP]);
            dst_block[j] = alpha_ * src_block[j] + beta_ * dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
    else
    {
        Index j_max = (nelems-i+BLOCK_STEP-1) / BLOCK_STEP;
        for(Index j = 0; j < j_max; ++j)
        {
            src_block[j] = static_cast<Y>(src1[i+j*BLOCK_STEP]);
            dst_block[j] = static_cast<Y>(src2[i+j*BLOCK_STEP]);
            dst_block[j] = alpha_ * src_block[j] + beta_ * dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
}

template<typename T>
void cuda(
    cudaStream_t stream,
    Index nelems,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add two buffers with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i] = alpha*src1[i] + beta*src2[i]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and src2[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha or beta being
 * zero, use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination tensor
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(
        nelems, alpha, src1, beta, src2, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src1, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src1, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src1, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *src1, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add
