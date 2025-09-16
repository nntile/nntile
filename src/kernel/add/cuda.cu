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

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T *src1, Scalar beta_,
        const T *src2, T *dst)
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src1[i] + beta*src2[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src1 tensor
 * @param[in] src1: Source tensor
 * @param[in] beta_: Scalar multiplier for the scr2 tensor
 * @param[in] src2: Source tensor
 * @param[out] dst: Destination of the add operation
 * */
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    Y dst_block[LOOP];
    Y src_block[LOOP];
    Y alpha = Y{alpha_};
    Y beta = Y{beta_};
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < LOOP; ++j)
        {
            dst_block[j] = static_cast<Y>(src1[i+j*BLOCK_STEP]);
            src_block[j] = static_cast<Y>(src2[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*src_block[j] + beta*dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
    else
    {
        int j_max = (nelems-i+BLOCK_STEP-1) / BLOCK_STEP;
        for(int j = 0; j < j_max; ++j)
        {
            dst_block[j] = static_cast<Y>(src1[i+j*BLOCK_STEP]);
            src_block[j] = static_cast<Y>(src2[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*src_block[j] + beta*dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha_, const T *src1_,
        Scalar beta_, const T *src2_, T *dst_)
    noexcept
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src1[i] + beta*src2[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src1 tensor
 * @param[in] src1_: Source tensor
 * @param[in] beta_: Scalar multiplier for the src2 tensor
 * @param[in] src2_: Source tensor
 * @param[out] dst_: Destination of the add operation
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems, alpha_,
            src1_, beta_, src2_, dst_);
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
