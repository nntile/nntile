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
void cuda_kernel(Index nelems, Scalar alpha_, const T *src, Scalar beta_, T *dst)
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the add operation
 * */
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    __shared__ T src1_block[BLOCK];
    __shared__ T src2_block[BLOCK];
    Y alpha = Y{alpha_};
    Y beta = Y{beta_};
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src[i+j];
        }
        __syncthreads();
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            src2_block[threadIdx.x+j] = dst[i+j];
        }
        __syncthreads();
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(
                    alpha*static_cast<Y>(src1_block[threadIdx.x+j]) +
                    beta*static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
    else
    {
        for(int j = 0; j < nelems-blockIdx.x*BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src[i+j];
        }
        __syncthreads();
        for(int j = 0; j < nelems-blockIdx.x*BLOCK; j += BLOCK_STEP)
        {
            src2_block[threadIdx.x+j] = dst[i+j];
        }
        __syncthreads();
        for(int j = 0; j < nelems-blockIdx.x*BLOCK; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(
                    alpha*static_cast<Y>(src1_block[threadIdx.x+j]) +
                    beta*static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha_, const T *src_,
        Scalar beta_, T *dst_)
    noexcept
//! Add two buffers on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 * where alpha and beta are non-zero scalars.
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add operation
 * */
{
    dim3 threads(128);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 8>)<<<blocks, threads, 0, stream>>>(nelems, alpha_,
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

} // namespace nntile::kernel::add
