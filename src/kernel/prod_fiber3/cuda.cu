/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/prod_fiber3/cuda.cu
 * Per-element multiplication of a tensor by a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/prod_fiber3/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::prod_fiber3
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Scalar alpha_,
        const T * __restrict__ src1, const T * __restrict__ src2,
        T * __restrict__ dst)
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i0 < m and i1 < n and i2 < k)
    {
        const Y src1_val = alpha * Y{src1[i2]};
        // Input fiber to be used
        const T *src2_fiber = src2 + (i1*k+i2)*m;
        // Output fiber to be updated
        T *dst_fiber = dst + (i1*k+i2)*m;
        // Update output value
        dst_fiber[i0] = T{src1_val * Y{src2_fiber[i0]}};
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, T *dst)
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[0,l,j] = alpha * src1[l] * src2[0,l,j]
 *
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous 1-by-k-by-n array
 * @param[out] dst: Output contiguous 1-by-k-by-n array
 * */
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    __shared__ Y src1_block[BLOCK_ROW];
    __shared__ T src2_block[BLOCK_ROW][BLOCK_COL+1];
    Index src1_l = dst_block_l*BLOCK_ROW + threadIdx.x;
    if(src1_l < k and threadIdx.x < BLOCK_ROW)
    {
        src1_block[threadIdx.x] = alpha * static_cast<Y>(src1[src1_l]);
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        const T *src2_fiber = src2 + global_dst_l + global_dst_j*k;
        T *dst_fiber = dst + global_dst_l + global_dst_j*k;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                src2_block[dst_l][c] = src2_fiber[c*k];
            }
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(src1_block[dst_l] *
                        static_cast<Y>(src2_block[dst_l][c]));
            }
        }
        else
        {
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                src2_block[dst_l][c] = src2_fiber[c*k];
            }
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(src1_block[dst_l] *
                        static_cast<Y>(src2_block[dst_l][c]));
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, T *dst)
    noexcept
//! Per-element product of a tensor and a broadcasted fiber on CPU
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    // Custom version of code for special m==1 case
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, src2, dst);
    }
    // Generic case
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src1,
                src2, dst);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src1, const fp16_t *src2, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::prod_fiber3
