/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_slice/cuda.cu
 * Sums over fibers into a slice of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sum_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sum_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    Y alpha{alpha_};
    Y beta{beta_};
    constexpr Y zero{0.0};
    if(i0 < m and i1 < n)
    {
        // Pointer to a corresponding fiber of the source array src
        const T *src_fiber = src + i1*mk + i0;
        // Init sum over the fiber
        Y sum{zero};
        // Cycle over fiber elements and accumulate the sum
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            sum += Y{src_fiber[i2*m]};
        }
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y], sum);
        __syncthreads();
        // Update output value
        if(i2_start == 0)
        {
            // Output value
            T &result = dst[i1*m+i0];
            sum = block_sum[threadIdx.x+blockDim.x*threadIdx.y];
            if(beta == zero)
            {
                result = T{alpha * sum};
            }
            else
            {
                result = T{beta * Y{result} + alpha * sum};
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided 1-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[0,j] = beta*dst[0,j] + alpha*sum(src[0,:,j])
 *
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src: Input contiguous 1-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous 1-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    volatile __shared__ Y dst_block[BLOCK_ROW_STEP];
    Y dst_val = 0.0;
    // Pointer to a corresponding fiber of the input arrays
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        const T *src_fiber = src + src_l + blockIdx.x*k;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y val = static_cast<Y>(src_fiber[c]);
            dst_val += val;
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index src_l = threadIdx.x + src_l_block_end;
    const T *src_fiber = src + blockIdx.x*k;
    for(Index c = src_l; c < k; c += BLOCK_ROW_STEP)
    {
        Y val = static_cast<Y>(src_fiber[c]);
        dst_val += val;
    }
    // Put calculated value into shared memory
    dst_block[threadIdx.x] = alpha * dst_val;
    __syncthreads();
    // Inter-warp reduction
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            dst_block[threadIdx.x] += dst_block[threadIdx.x+c];
        }
        __syncthreads();
    }
    // Reduction within a single warp
    if(threadIdx.x < 32)
    {
        for(int c = 32; c > 0; c >>= 1)
        {
            dst_block[threadIdx.x] += dst_block[threadIdx.x+c];
        }
    }
    // Write output
    if(threadIdx.x == 0)
    {
        if(beta == 0.0)
        {
            dst[blockIdx.x] = static_cast<T>(static_cast<Y>(dst_block[0]));
        }
        else
        {
            dst_val = beta * static_cast<Y>(dst[blockIdx.x]);
            dst[blockIdx.x] = static_cast<T>(dst_val + dst_block[0]);
        }
    }
}

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
static __global__
void cuda_kernel2(Index m, Index n, Index k, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
{
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    Index griddim_m = (m+BLOCK_M-1) / BLOCK_M;
    Index block_i = blockIdx.x % griddim_m;
    Index block_j = blockIdx.x / griddim_m;
    Index i = block_i * BLOCK_M;
    Index j = block_j * BLOCK_N;
    __shared__ T src_block[BLOCK_M][BLOCK_K][BLOCK_N];
    __shared__ Y dst_block[BLOCK_M][BLOCK_N];
    // Init dst_block
    for(Index ii = threadIdx.x; ii < BLOCK_M; ii += blockDim.x)
    {
        for(Index jj = 0; jj < BLOCK_N; ++jj)
        {
            dst_block[ii][jj] = 0.0;
        }
    }
    // Proceed by blocks in dimension of k
    for(Index l = 0; l < k; l+= BLOCK_K)
    {
        // Read block of src
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                const T *src_fiber = src + ii + jj*m*k;
                for(Index ll = l; ll < ::min(k, l+BLOCK_K); ++ll)
                {
                    src_block[ii-i][ll-l][jj-j] = src_fiber[ll*m];
                }
            }
        }
        // Now proceed with data in shared memory
        __syncthreads();
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                Y dst_val = 0.0;
                for(Index ll = l; ll < ::min(k, l+BLOCK_K); ++ll)
                {
                    T val = src_block[ii-i][ll-l][jj-j];
                    dst_val += static_cast<Y>(val);
                }
                dst_block[ii-i][jj-j] += dst_val;
            }
        }
    }
    // Update output
    if(beta == 0.0)
    {
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                Y val = alpha * dst_block[ii-i][jj-j];
                dst[ii+jj*m] = static_cast<T>(val);
            }
        }
    }
    else
    {
        for(Index ii = i+threadIdx.x; ii < ::min(m, i+BLOCK_M);
                ii += blockDim.x)
        {
            for(Index jj = j; jj < ::min(n, j+BLOCK_N); ++jj)
            {
                Y val = alpha * dst_block[ii-i][jj-j];
                val += beta * static_cast<Y>(dst[ii+jj*m]);
                dst[ii+jj*m] = static_cast<T>(val);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept
//! Sums over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array computes sums over fibers
 * along second axis with k elements, resulting in m-by-n output slice.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum(src[i,:,j])
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums over fibers along middle axis.
 * */
{
    // Both source and destination are Fortran-contiguous
    // Separate case for m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(n);
        (cuda_kernel_m1<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src, beta, dst);
    }
    // Case of small k
    else if(k < 1024)
    {
        dim3 threads(256);
        dim3 blocks(((m+255)/256) * n);
        (cuda_kernel2<T, 256, 1, 4>)<<<blocks, threads, 0, stream>>>(m, n, k,
                alpha, src, beta, dst);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src, beta, dst);
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

} // namespace nntile::kernel::sum_slice
