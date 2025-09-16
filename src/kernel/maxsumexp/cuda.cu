/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/maxsumexp/cuda.cu
 * Max and sum of exponents of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/maxsumexp/cuda.hh"
#include <cmath>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::maxsumexp
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index m_per_block, Index n, Index n_per_block,
        Index k, Index mk, const T * __restrict__ src,
        T * __restrict__ maxsumexp)
{
    Index i1_block = blockIdx.y, i2_block = blockIdx.z,
          i0_start = threadIdx.x, i0_step = blockDim.x;

    using Y = typename T::repr_t;
    constexpr Y zero = 0.0, one = 1.0;
    if(i0_start < k)
    {
        for(Index i1 = i1_block*m_per_block;
                i1 < (i1_block+1)*m_per_block and i1 < m; ++i1)
        {
            for(Index i2 = i2_block*n_per_block;
                    i2 < (i2_block+1)*n_per_block and i2 < n; ++i2)
            {
                // Get max and sum of exponents of a corresponding slice
                const T *src_slice = src + i2*mk + i1;
                // Init max and sum
                Y max_val = Y{src_slice[i0_start*m]};
                Y sum_val = one;
                // Cycle over slice of input buffer
                for(Index i0 = i0_start+i0_step; i0 < k; i0 += i0_step)
                {
                    // Read value from source
                    Y val = Y{src_slice[i0*m]};
                    // Ignore -inf value, which comes from mask
                    if(::isinf(val))
                    {
                        continue;
                    }
                    // Update max and sum of exponents
                    if(max_val < val)
                    {
                        sum_val = sum_val*(::exp(max_val-val)) + one;
                        max_val = val;
                    }
                    else
                    {
                        sum_val += ::exp(val-max_val);
                    }
                }
                // Per-block of threads max and sum of exponents
                volatile __shared__ Y block_max_val;
                __shared__ Y block_sum_val;
                // Init shared values in the i0_start==0 thread
                if(i0_start == 0)
                {
                    block_max_val = max_val;
                    block_sum_val = zero;
                }
                // Other threads wait until initialization is done
                __syncthreads();
                // Update max at first
                while(block_max_val < max_val)
                {
                    block_max_val = max_val;
                }
                // Sync with all other threads to get per-block max finally
                __syncthreads();
                // Accumulate per-block sum of finite values
                if(not ::isinf(max_val))
                {
                    sum_val *= ::exp(max_val - block_max_val);
                    atomicAdd(&block_sum_val, sum_val);
                }
                __syncthreads();
                // Update output iff per-block sum is not zero
                if(i0_start == 0 and block_sum_val > 0)
                {
                    // Get per-block max and sum of exponents into local
                    // variables
                    max_val = block_max_val;
                    sum_val = block_sum_val;
                    Index dst_offset = i1 + i2*m;
                    // Now max_val is finite, we need to accumulate sum of
                    // exponents with the data in global memory
                    Y max_output;
                    Y sum_output = Y{maxsumexp[2*dst_offset+1]};
                    // If data was not yet initialised, just overwrite it
                    if(sum_output == zero)
                    {
                        max_output = max_val;
                        sum_output = sum_val;
                    }
                    // Accumulate otherwise
                    else
                    {
                        max_output = Y{maxsumexp[2*dst_offset]};
                        if(max_val < max_output)
                        {
                            sum_val *= ::exp(max_val - max_output);
                        }
                        else
                        {
                            sum_output *= ::exp(max_output - max_val);
                            max_output = max_val;
                        }
                        sum_output += sum_val;
                    }
                    maxsumexp[2*dst_offset] = T{max_output};
                    maxsumexp[2*dst_offset+1] = T{sum_output};
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, const T *src, T *dst)
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    constexpr Y one = 1.0;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    T src_val[LOOP];
    volatile __shared__ Y dst_block_max[BLOCK_ROW_STEP];
    volatile __shared__ Y dst_block_sumexp[BLOCK_ROW_STEP];
    Y dst_max=-INFINITY, dst_sumexp=0.0;
    // Pointer to a corresponding fiber of the input arrays
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        const T *src_fiber = src + src_l + blockIdx.x*k;
        for(int c = 0; c < LOOP; ++c)
        {
            src_val[c] = src_fiber[c*BLOCK_ROW_STEP];
        }
        for(int c = 0; c < LOOP; ++c)
        {
            Y val = static_cast<Y>(src_val[c]);
            if(not ::isinf(val))
            {
                if(dst_max < val)
                {
                    dst_sumexp = one + ::exp(dst_max-val)*dst_sumexp;
                    dst_max = val;
                }
                else
                {
                    dst_sumexp += ::exp(val-dst_max);
                }
            }
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index src_l = threadIdx.x + src_l_block_end;
    const T *src_fiber = src + src_l + blockIdx.x*k;
    int c_max = (k-src_l+BLOCK_ROW_STEP-1) / BLOCK_ROW_STEP;
    for(int c = 0; c < c_max; ++c)
    {
        src_val[c] = src_fiber[c*BLOCK_ROW_STEP];
    }
    for(int c = 0; c < c_max; ++c)
    {
        Y val = static_cast<Y>(src_val[c]);
        if(not ::isinf(val))
        {
            if(dst_max < val)
            {
                dst_sumexp = one + ::exp(dst_max-val)*dst_sumexp;
                dst_max = val;
            }
            else
            {
                dst_sumexp += ::exp(val-dst_max);
            }
        }
    }
    // Put calculated value into shared memory
    dst_block_max[threadIdx.x] = dst_max;
    dst_block_sumexp[threadIdx.x] = dst_sumexp;
    __syncthreads();
    // Inter-warp reduction
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            Y max2 = dst_block_max[threadIdx.x+c];
            if(not ::isinf(max2))
            {
                volatile Y &max = dst_block_max[threadIdx.x];
                volatile Y &sumexp = dst_block_sumexp[threadIdx.x];
                Y sumexp2 = dst_block_sumexp[threadIdx.x+c];
                if(max < max2)
                {
                    sumexp = sumexp2 + ::exp(max-max2)*sumexp;
                    max = max2;
                }
                else
                {
                    sumexp += ::exp(max2-max) * sumexp2;
                }
            }
        }
        __syncthreads();
    }
    // Reduction within a single warp
    if(threadIdx.x < 32)
    {
        for(int c = 32; c > 0; c >>= 1)
        {
            Y max2 = dst_block_max[threadIdx.x+c];
            if(not ::isinf(max2))
            {
                volatile Y &max = dst_block_max[threadIdx.x];
                volatile Y &sumexp = dst_block_sumexp[threadIdx.x];
                Y sumexp2 = dst_block_sumexp[threadIdx.x+c];
                if(max < max2)
                {
                    sumexp = sumexp2 + ::exp(max-max2)*sumexp;
                    max = max2;
                }
                else
                {
                    sumexp += ::exp(max2-max) * sumexp2;
                }
            }
        }
    }
    // Write output
    if(threadIdx.x == 0)
    {
        Y max2 = dst_block_max[0];
        Y sumexp2 = dst_block_sumexp[0];
        if(not ::isinf(max2))
        {
            Y max = static_cast<Y>(dst[2*blockIdx.x]);
            Y sumexp = static_cast<Y>(dst[2*blockIdx.x+1]);
            if(sumexp == 0.0)
            {
                sumexp = sumexp2;
                max = max2;
            }
            else if(max < max2)
            {
                sumexp = sumexp2 + ::exp(max-max2)*sumexp;
                max = max2;
            }
            else
            {
                sumexp += ::exp(max2-max) * sumexp2;
            }
            dst[2*blockIdx.x] = static_cast<T>(max);
            dst[2*blockIdx.x+1] = static_cast<T>(sumexp);
        }
    }
}

template <typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
          T *maxsumexp)
    noexcept
{
    // Both source and destination are Fortran-contiguous
    // Custom case m==1
    if(m == 1)
    {
        if(k <= 1024)
        {
            dim3 threads(64);
            dim3 blocks(n);
            (cuda_kernel_m1<T, 1024, 16>)<<<blocks, threads, 0, stream>>>(n,
                    k, src, maxsumexp);
        }
        else if(k <= 2048)
        {
            dim3 threads(128);
            dim3 blocks(n);
            (cuda_kernel_m1<T, 2048, 16>)<<<blocks, threads, 0, stream>>>(n,
                    k, src, maxsumexp);
        }
        else
        {
            dim3 threads(256);
            dim3 blocks(n);
            (cuda_kernel_m1<T, 4096, 16>)<<<blocks, threads, 0, stream>>>(n,
                    k, src, maxsumexp);
        }
    }
    else
    {
        dim3 threads(32, 1, 1), blocks(1, m, n);
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
                n_per_block, k, m*k, src, maxsumexp);
    }
}

template void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, fp32_t *maxsumexp)
    noexcept;

template void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, fp64_t *maxsumexp)
    noexcept;

template void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *src, bf16_t *maxsumexp)
    noexcept;

template void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp16_t *src, fp16_t *maxsumexp)
    noexcept;

} // namespace nntile::kernel::maxsumexp
