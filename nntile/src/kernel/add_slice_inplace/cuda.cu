/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_slice_inplace/cuda.cu
 * Per-element addition of a tensor and a broadcasted slice on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/add_slice_inplace/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_slice_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
//! Generic implementation of the add_slice_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_slice_inplace::cuda
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y beta{beta_};
    const Y alpha{alpha_};
    constexpr Y zero{0.0};
    if(i2 < k and i1 < n and i0 < m)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + i1*mk + i0;
        // Value to add to the output fiber
        const Y src_val = Y{alpha} * Y{src[i1*m+i0]};
        // Read value from the output
        T &dst_val = dst_fiber[i2*m];
        // And update it
        dst_val = T{beta * Y{dst_val} + src_val};
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Generic implementation of the add_slice_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_slice_inplace::cuda
 * */
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y beta{beta_};
    const Y alpha{alpha_};
    constexpr Y zero{0.0};
    __shared__ Y src_block[BLOCK_COL];
    __shared__ T dst_block[BLOCK_ROW][BLOCK_COL];
    Index src_j = dst_block_j*BLOCK_COL + threadIdx.x;
    if(src_j < n and threadIdx.x < BLOCK_COL)
    {
        src_block[threadIdx.x] = alpha * static_cast<Y>(src[src_j]);
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        // Pointer to a corresponding fiber of the output array dst
        T *dst_fiber = dst + global_dst_l + global_dst_j*k;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = dst_fiber[c*k];
            }
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(
                        beta*static_cast<Y>(dst_block[dst_l][c]) +
                        src_block[c]);
            }
        }
        else
        {
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = dst_fiber[c*k];
            }
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(
                        beta*static_cast<Y>(dst_block[dst_l][c]) +
                        src_block[c]);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src_, Scalar beta, T *dst_)
    noexcept
//! Add a broadcasted slice to a tensor with optional scaling inplace on CUDA
/*! Performs the following operation:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[i,j]
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[i,j] is NaN, then dst[i,l,j] will be NaN.
 * If beta is zero and dst[i,l,j] is NaN, then dst[i,l,j] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace instead, and in a case of beta being zero,
 * use nntile::kernel::scale_slice instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale_slice
 * @see nntile::kernel::clear
 *
 * @param[in] m: Size of the first mode of src and dst tensors
 * @param[in] n: Size of the last mode of src and dst tensors
 * @param[in] k: Size of the middle mode of dst tensor
 * @param[in] alpha_: Scalar factor for src
 * @param[in] src: Input contiguous m-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    // Custom version of code for special m==1 case
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src_, beta, dst_);
    }
    // Generic case
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src_, beta, dst_);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::add_slice_inplace
