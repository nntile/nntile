/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_slice/cuda.cu
 * Sums over fibers into a slice of a product of buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sumprod_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sumprod_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src1, const T *src2, Scalar beta_, T *dst)
//! Sums over fibers into a slice of a product of two tensors
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding fibers along second axis with k
 * elements, resulting in m-by-n output array dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum_l(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis of per-element products of src1 and src2.
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero{0.};
    const Y alpha{alpha_};
    const Y beta{beta_};
    if(i0 < m and i1 < n)
    {
        // Get corresponding fibers of both sources
        const T *src1_fiber = src1 + i1*mk + i0;
        const T *src2_fiber = src2 + i1*mk + i0;
        // Init sum of product of the fibers
        Y sum = zero;
        // Cycle over fibers of inputs
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            // Update sum
            sum += Y{src1_fiber[i2*m]} * Y{src2_fiber[i2*m]};
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

template<typename T, int BLOCK>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, Scalar beta_, T *dst)
//! Sums over fibers into a slice of a product of two tensors
/*! For two provided 1-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding fibers along second axis with k
 * elements, resulting in 1-by-n output array dst.
 * Mnemonically, the following operations are performed:
 *      dst[0,j] = beta*dst[0,j] + alpha*sum_l(src1[0,l,j] * src2[0,l,j])
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis of per-element products of src1 and src2.
 * */
{
    Index src_l = threadIdx.x;
    Index src_j = blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    __shared__ T src1_block[BLOCK][BLOCK+1];
    __shared__ T src2_block[BLOCK][BLOCK+1];
    __shared__ Y dst_block[BLOCK];
    dst_block[threadIdx.x] = 0.0;
    T *dst_fiber = dst + src_j;
    if(n-src_j >= BLOCK)
    {
        while(src_l-threadIdx.x+BLOCK < k)
        {
            // Pointer to a corresponding fiber of the input arrays
            const T *src1_fiber = src1 + src_l + src_j*k;
            const T *src2_fiber = src2 + src_l + src_j*k;
            for(int c = 0; c < BLOCK; ++c)
            {
                src1_block[threadIdx.x][c] = src1_fiber[c*k];
            }
            for(int c = 0; c < BLOCK; ++c)
            {
                src2_block[threadIdx.x][c] = src2_fiber[c*k];
            }
            __syncthreads();
            for(int c = 0; c < BLOCK; ++c)
            {
                dst_block[threadIdx.x] +=
                    static_cast<Y>(src1_block[c][threadIdx.x]) *
                    static_cast<Y>(src2_block[c][threadIdx.x]);
            }
            src_l += BLOCK;
        }
        // Pointer to a corresponding fiber of the input arrays
        const T *src1_fiber = src1 + src_l + src_j*k;
        const T *src2_fiber = src2 + src_l + src_j*k;
        if(src_l < k)
        {
            for(int c = 0; c < BLOCK; ++c)
            {
                src1_block[threadIdx.x][c] = src1_fiber[c*k];
            }
            for(int c = 0; c < BLOCK; ++c)
            {
                src2_block[threadIdx.x][c] = src2_fiber[c*k];
            }
        }
        __syncthreads();
        for(int c = 0; c < k+threadIdx.x-src_l; ++c)
        {
            dst_block[threadIdx.x] +=
                static_cast<Y>(src1_block[c][threadIdx.x]) *
                static_cast<Y>(src2_block[c][threadIdx.x]);
        }
        if(beta == 0.0)
        {
            dst_fiber[threadIdx.x] = static_cast<T>(
                    alpha*dst_block[threadIdx.x]);
        }
        else
        {
            dst_fiber[threadIdx.x] = static_cast<T>(
                    beta*static_cast<Y>(dst_fiber[threadIdx.x]) +
                    alpha*dst_block[threadIdx.x]);
        }
    }
    else
    {
        while(src_l-threadIdx.x+BLOCK < k)
        {
            // Pointer to a corresponding fiber of the input arrays
            const T *src1_fiber = src1 + src_l + src_j*k;
            const T *src2_fiber = src2 + src_l + src_j*k;
            for(int c = 0; c < n-src_j; ++c)
            {
                src1_block[threadIdx.x][c] = src1_fiber[c*k];
            }
            for(int c = 0; c < n-src_j; ++c)
            {
                src2_block[threadIdx.x][c] = src2_fiber[c*k];
            }
            __syncthreads();
            if(threadIdx.x < n-src_j)
            {
                for(int c = 0; c < BLOCK; ++c)
                {
                    static_cast<Y>(src1_block[c][threadIdx.x]) *
                    static_cast<Y>(src2_block[c][threadIdx.x]);
                }
            }
            src_l += BLOCK;
        }
        // Pointer to a corresponding fiber of the input arrays
        const T *src1_fiber = src1 + src_l + src_j*k;
        const T *src2_fiber = src2 + src_l + src_j*k;
        if(src_l < k)
        {
            for(int c = 0; c < n-src_j; ++c)
            {
                src1_block[threadIdx.x][c] = src1_fiber[c*k];
            }
            for(int c = 0; c < n-src_j; ++c)
            {
                src2_block[threadIdx.x][c] = src2_fiber[c*k];
            }
        }
        __syncthreads();
        if(threadIdx.x < n-src_j)
        {
            for(int c = 0; c < k+threadIdx.x-src_l; ++c)
            {
                dst_block[threadIdx.x] +=
                    static_cast<Y>(src1_block[c][threadIdx.x]) *
                    static_cast<Y>(src2_block[c][threadIdx.x]);
            }
        }
        if(beta == 0.0)
        {
            if(threadIdx.x < n-src_j)
            {
                dst_fiber[threadIdx.x] = static_cast<T>(
                        alpha*dst_block[threadIdx.x]);
            }
        }
        else
        {
            if(threadIdx.x < n-src_j)
            {
                dst_fiber[threadIdx.x] = static_cast<T>(
                        beta*static_cast<Y>(dst_fiber[threadIdx.x]) +
                        alpha*dst_block[threadIdx.x]);
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, Scalar beta, T *dst)
    noexcept
//! Sums over fibers into a slice of a product of two tensors
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding fibers along second axis with k
 * elements, resulting in m-by-n output array dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = beta*dst[i,j] + alpha*sum_l(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst
 * @param[in] n: Size of the last mode of src1, src2 and dst
 * @param[in] k: Size of the middle mode of src1 and src2 arrays
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous m-by-n array, that accumulates
 *      sums along middle axis of per-element products of src1 and src2.
 * */
{
    // Both source and destination are Fortran-contiguous
    // Separate case for m==1
    if(m == 1)
    {
        dim3 threads(64);
        dim3 blocks((n+63)/64);
        //printf("n=%ld\n", n);
        (cuda_kernel_m1<T, 64>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, src2, beta, dst);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8), 16);
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src1, src2, beta, dst);
        printf("m=%ld,n=%ld,k=%ld,alpha=%f,beta=%f\n", m, n, k, alpha, beta);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, Scalar beta, fp32_t *sum_dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_fast_tf32_t *src1, const fp32_fast_tf32_t *src2, Scalar beta, fp32_fast_tf32_t *sum_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, Scalar beta, fp64_t *sum_dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, Scalar beta, bf16_t *sum_dst)
    noexcept;

} // namespace nntile::kernel::sumprod_slice
