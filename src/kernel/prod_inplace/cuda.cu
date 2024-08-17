/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_inplace/cuda.cu
 * Per-element product of two buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/prod_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::prod_inplace
{

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    __shared__ T src1_block[BLOCK];
    __shared__ T src2_block[BLOCK];
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src[i+j];
            src2_block[threadIdx.x+j] = dst[i+j];
        }
        for(int j = 0; j < BLOCK; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(
                    static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
    else
    {
        for(int j = 0; j < nelems-blockIdx.x*BLOCK; j += BLOCK_STEP)
        {
            src1_block[threadIdx.x+j] = src[i+j];
            src2_block[threadIdx.x+j] = dst[i+j];
        }
        for(int j = 0; j < nelems-blockIdx.x*BLOCK; j += BLOCK_STEP)
        {
            dst[i+j] = static_cast<T>(
                    static_cast<Y>(src1_block[threadIdx.x+j]) *
                    static_cast<Y>(src2_block[threadIdx.x+j]));
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! Per-element product of two buffers
/*! One of the buffers serves as output
 *
 * @param[in] nelems: Number of elements in both buffers
 * @param[in] src: Input buffer
 * @param[inout] dst: Input buffers that contains output in the end
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *src,
        fp32_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index nelems,
        const fp32_fast_tf32_t *src, fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *src,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *src,
        bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::prod_inplace
