/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/softmax/cuda.cu
 * Softmax operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/softmax/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::softmax
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index m_per_block, Index n, Index n_per_block,
        Index k, const T * __restrict__ maxsumexp,
        const T * __restrict__ src, Scalar alpha, T * __restrict__ dst_)
{
    Index i0_block = blockIdx.y, i1_block = blockIdx.z,
          i2_start = threadIdx.x, i2_step = blockDim.x;

    using Y = typename T::repr_t;
    for(Index i0 = i0_block*m_per_block;
            i0 < (i0_block+1)*m_per_block and i0 < m; ++i0)
    {
        for(Index i1 = i1_block*n_per_block;
                i1 < (i1_block+1)*n_per_block and i1 < n; ++i1)
        {
            // Offset in memory for src and dst
            Index src_dst_offset = i1*k*m + i0;
            // Input and output fiber/slice
            const T *src_slice = src + src_dst_offset;
            T *dst_slice = dst_ + src_dst_offset;
            // Max and sum of exponents
            __shared__ Y max, sum;
            if(i2_start == 0)
            {
                Index maxsumexp_offset = m*i1 + i0;
                max = Y{maxsumexp[2*maxsumexp_offset]};
                sum = Y{maxsumexp[2*maxsumexp_offset+1]};
            }
            __syncthreads();
            for(Index i2 = i2_start; i2 < k; i2 += i2_step)
            {
                // Value-to-update
                Y val = Y{src_slice[i2*m]};
                // Update value
                if(not ::isinf(val))
                {
                    dst_slice[i2*m] = Y{alpha} * ::exp(val-max) / sum;
                }
                else
                {
                    dst_slice[i2*m] = 0.0;
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, const T *maxsumexp, const T *src,
        Scalar alpha_, T *dst)
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    constexpr Y zero{0.0};
    __shared__ Y max_block[BLOCK_COL];
    __shared__ Y sumexp_block[BLOCK_COL];
    __shared__ T dst_block[BLOCK_ROW][BLOCK_COL];
    Index maxsumexp_j = dst_block_j*BLOCK_COL + threadIdx.x;
    if(maxsumexp_j < n and threadIdx.x < BLOCK_COL)
    {
        max_block[threadIdx.x] = static_cast<Y>(maxsumexp[2*maxsumexp_j]);
        sumexp_block[threadIdx.x] = static_cast<Y>(maxsumexp[2*maxsumexp_j+1]);
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        // Pointer to a corresponding fiber of the output array dst
        const T *src_fiber = src + global_dst_l + global_dst_j*k;
        T *dst_fiber = dst + global_dst_l + global_dst_j*k;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = src_fiber[c*k];
            }
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                Y val = static_cast<Y>(dst_block[dst_l][c]);
                if(not ::isinf(val))
                {
                    val = alpha * ::exp(val-max_block[c]) / sumexp_block[c];
                    dst_fiber[c*k] = static_cast<T>(val);
                }
                else
                {
                    dst_fiber[c*k] = static_cast<T>(0.0);
                }
            }
        }
        else
        {
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = src_fiber[c*k];
            }
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                Y val = static_cast<Y>(dst_block[dst_l][c]);
                if(not ::isinf(val))
                {
                    val = alpha * ::exp(val-max_block[c]) / sumexp_block[c];
                    dst_fiber[c*k] = static_cast<T>(val);
                }
                else
                {
                    dst_fiber[c*k] = static_cast<T>(0.0);
                }
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp_,
        const T *src_, Scalar alpha, T *dst_)
    noexcept
//! Softmax of a buffer along middle axis
/*!
 *
 * @param[in] m: Size of the first mode of dst and sumexp arrays
 * @param[in] n: Size of the last mode of dst and sumexp arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] maxsumexp_: Maximums and sums of exponents of slices
 * @param[in] src_: The source input data
 * @param[in] alpha: Scalar multipler for the output
 * @param[in] dst_: Contiguous output array
 * */
{
    // Both source and destination are Fortran-contiguous
    // Custom case m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0, stream>>>(n, k,
                maxsumexp_, src_, alpha, dst_);
    }
    else
    {
        dim3 threads(32, 1, 1);
        dim3 blocks(1, m, n);
        Index m_per_block = 1, n_per_block = 1;
        if(m > 65535)
        {
            m_per_block = (m+65534) / 65535;
            blocks.y = (m+m_per_block-1) / m_per_block;
        }
        if(n > 65535)
        {
            n_per_block = (n+65534) / 65535;
            blocks.z = (n+n_per_block-1) / n_per_block;
        }
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, m_per_block, n,
                n_per_block, k, maxsumexp_, src_, alpha, dst_);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *maxsumexp, const fp32_t *src, Scalar alpha, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *maxsumexp, const fp64_t *src, Scalar alpha, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *maxsumexp, const bf16_t *src, Scalar alpha, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp16_t *maxsumexp, const fp16_t *src, Scalar alpha, fp16_t *dst)
    noexcept;

} // namespace nntile::kernel::softmax
