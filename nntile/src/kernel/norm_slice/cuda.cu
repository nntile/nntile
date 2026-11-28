/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/norm_slice/cuda.cu
 * Euclidean norms of fibers into a slice of a buffer on CUDA (out-of-place version)
 *
 * @version 2.0.0
 * */

#include "nntile/kernel/norm_slice/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::norm_slice
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src1, Scalar beta_, const T *src2, T *dst)
//! Euclidean norms over fibers along middle axis into a slice of a tensor (out-of-place version)
/*! For a provided m-by-k-by-n input array src1 compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*src2[i,j], alpha*norm(src1[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst arrays
 * @param[in] n: Size of the last mode of src1, src2 and dst arrays
 * @param[in] k: Size of the middle mode of src1 array
 * @param[in] alpha_: Scaling factor for src1
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for src2
 * @param[in] src2: Input contiguous m-by-n array
 * @param[out] dst: Output contiguous m-by-n array, that contains norms
 *      along middle axis combined with src2 values.
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    Index idx = threadIdx.x+blockDim.x*threadIdx.y;
    using Y = typename T::repr_t;
    const Y alpha = static_cast<Y>(alpha_), beta = static_cast<Y>(beta_);
    constexpr Y zero = 0.0;
    if(i0 < m and i1 < n)
    {
        // Pointer to a corresponding fiber of the source array src1
        const T *src1_fiber = src1 + i1*mk + i0;
        // Init sum over the fiber
        Y sum = zero;
        // Cycle over fiber elements and accumulate the sum
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            Y src1_val = static_cast<Y>(src1_fiber[i2*m]);
            sum = ::hypot(sum, src1_val);
        }
        volatile __shared__ Y block_max[64];
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_max[idx] = sum;
            block_sum[idx] = zero;
        }
        __syncthreads();
        while(block_max[idx] < sum)
        {
            block_max[idx] = sum;
        }
        __syncthreads();
        if(block_max[idx] > 0)
        {
            sum /= block_max[idx];
            atomicAdd(&block_sum[idx], sum*sum);
        }
        __syncthreads();
        // Update output value
        if(i2_start == 0)
        {
            // Output value
            T &output_dst = dst[i1*m+i0];
            if(block_max[idx] > 0)
            {
                sum = block_max[idx];
                sum *= ::sqrt(block_sum[idx]);
                Y src2_val = beta * static_cast<Y>(src2[i1*m+i0]);
                if(beta == zero)
                {
                    output_dst = ::fabs(alpha) * sum;
                }
                else
                {
                    output_dst = ::hypot(src2_val, alpha*sum);
                }
            }
            // All-zero fiber: write the beta-combined value exactly like
            // the CPU kernel and the m==1 CUDA kernel of this family do
            // (0 for beta==0, fabs(beta*src2) otherwise). The old code
            // skipped the write entirely, leaving dst stale - divergent
            // from the CPU kernel on legal zero inputs (padding, masks)
            // and fatal for the piecewise hypot chain of the chunked
            // driver, whose continuation pieces anchor on the first
            // piece's output.
            else
            {
                if(beta == zero)
                {
                    output_dst = static_cast<T>(zero);
                }
                else
                {
                    output_dst = static_cast<T>(::fabs(
                            beta * static_cast<Y>(src2[i1*m+i0])));
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src1,
        Scalar beta_, const T *src2, T *dst)
//! Euclidean norms over fibers along middle axis into a slice of a tensor (out-of-place version)
/*! For a provided 1-by-k-by-n input array src1 compute norms of fibers
 * along second axis with k elements, resulting in 1-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[0,j] = hypot(beta*src2[0,j], alpha*norm(src1[0,:,j]))
 *
 * @param[in] n: Size of the last mode of src1, src2 and dst arrays
 * @param[in] k: Size of the middle mode of src1 array
 * @param[in] alpha_: Scaling factor for src1
 * @param[in] src1: Input contiguous 1-by-k-by-n array
 * @param[in] beta_: Scaling factor for src2
 * @param[in] src2: Input contiguous 1-by-n array
 * @param[out] dst: Output contiguous 1-by-n array, that contains norms
 *      along middle axis combined with src2 values.
 * */
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    const Y alpha = static_cast<Y>(alpha_), beta = static_cast<Y>(beta_);
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    volatile __shared__ Y dst_block[BLOCK_ROW_STEP];
    Y dst_val = 0.0;
    // Pointer to a corresponding fiber of the input arrays
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        const T *src1_fiber = src1 + src_l + blockIdx.x*k;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y src1_val = static_cast<Y>(src1_fiber[c]);
            dst_val = ::hypot(dst_val, src1_val);
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index src_l = threadIdx.x + src_l_block_end;
    const T *src1_fiber = src1 + blockIdx.x*k;
    for(Index c = src_l; c < k; c += BLOCK_ROW_STEP)
    {
        Y src1_val = static_cast<Y>(src1_fiber[c]);
        dst_val = ::hypot(dst_val, src1_val);
    }
    // Put calculated value into shared memory
    dst_block[threadIdx.x] = ::fabs(alpha) * dst_val;
    __syncthreads();
    // Inter-warp reduction
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            dst_block[threadIdx.x] = ::hypot(
                    dst_block[threadIdx.x], dst_block[threadIdx.x+c]);
        }
        __syncthreads();
    }
    // Reduction within a single warp
    if(threadIdx.x < 32)
    {
        for(int c = 32; c > 0; c >>= 1)
        {
            dst_block[threadIdx.x] = ::hypot(
                    dst_block[threadIdx.x], dst_block[threadIdx.x+c]);
        }
    }
    // Write output
    if(threadIdx.x == 0)
    {
        Y scaled_src2_val = beta * static_cast<Y>(src2[blockIdx.x]);
        if(beta == 0.0)
        {
            dst[blockIdx.x] = static_cast<Y>(dst_block[0]);
        }
        else
        {
            dst[blockIdx.x] = ::hypot(scaled_src2_val, dst_block[0]);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, Scalar beta, const T *src2, T *dst)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor (out-of-place version)
/*! For a provided m-by-k-by-n input array src1 compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*src2[i,j], alpha*norm(src1[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src1, src2 and dst arrays
 * @param[in] n: Size of the last mode of src1, src2 and dst arrays
 * @param[in] k: Size of the middle mode of src1 array
 * @param[in] alpha: Scaling factor for src1
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for src2
 * @param[in] src2: Input contiguous m-by-n array
 * @param[out] dst: Output contiguous m-by-n array, that contains norms
 *      along middle axis combined with src2 values.
 * */
{
    // Both source and destination are Fortran-contiguous
    // Separate case for m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(n);
        (cuda_kernel_m1<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, beta, src2, dst);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src1, beta, src2, dst);
    }
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        ChunkConst<T> src1, Scalar beta_, ChunkConst<T> src2, ChunkMut<T> dst)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    Index idx = threadIdx.x+blockDim.x*threadIdx.y;
    using Y = typename T::repr_t;
    const Y alpha = static_cast<Y>(alpha_), beta = static_cast<Y>(beta_);
    constexpr Y zero = 0.0;
    if(i0 < m and i1 < n)
    {
        // Flat offset of the corresponding fiber of src1
        Index src1_base = i1*mk + i0;
        Y sum = zero;
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            Y src1_val = static_cast<Y>(src1.at(src1_base + i2*m)[0]);
            sum = ::hypot(sum, src1_val);
        }
        volatile __shared__ Y block_max[64];
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_max[idx] = sum;
            block_sum[idx] = zero;
        }
        __syncthreads();
        while(block_max[idx] < sum)
        {
            block_max[idx] = sum;
        }
        __syncthreads();
        if(block_max[idx] > 0)
        {
            sum /= block_max[idx];
            atomicAdd(&block_sum[idx], sum*sum);
        }
        __syncthreads();
        if(i2_start == 0)
        {
            T *output_dst = dst.at(i1*m+i0);
            if(block_max[idx] > 0)
            {
                sum = block_max[idx];
                sum *= ::sqrt(block_sum[idx]);
                Y src2_val = beta * static_cast<Y>(src2.at(i1*m+i0)[0]);
                if(beta == zero)
                {
                    output_dst[0] = ::fabs(alpha) * sum;
                }
                else
                {
                    output_dst[0] = ::hypot(src2_val, alpha*sum);
                }
            }
            else
            {
                if(beta == zero)
                {
                    output_dst[0] = static_cast<T>(zero);
                }
                else
                {
                    output_dst[0] = static_cast<T>(::fabs(
                            beta * static_cast<Y>(src2.at(i1*m+i0)[0])));
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void chunked_cuda_kernel_m1(Index n, Index k, Scalar alpha_,
        ChunkConst<T> src1, Scalar beta_, ChunkConst<T> src2, ChunkMut<T> dst)
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    const Y alpha = static_cast<Y>(alpha_), beta = static_cast<Y>(beta_);
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    volatile __shared__ Y dst_block[BLOCK_ROW_STEP];
    Y dst_val = 0.0;
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        Index base = src_l + blockIdx.x*k;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y src1_val = static_cast<Y>(src1.at(base + c)[0]);
            dst_val = ::hypot(dst_val, src1_val);
        }
    }
    Index src_l = threadIdx.x + src_l_block_end;
    for(Index c = src_l; c < k; c += BLOCK_ROW_STEP)
    {
        Y src1_val = static_cast<Y>(src1.at(blockIdx.x*k + c)[0]);
        dst_val = ::hypot(dst_val, src1_val);
    }
    dst_block[threadIdx.x] = ::fabs(alpha) * dst_val;
    __syncthreads();
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            dst_block[threadIdx.x] = ::hypot(
                    dst_block[threadIdx.x], dst_block[threadIdx.x+c]);
        }
        __syncthreads();
    }
    if(threadIdx.x < 32)
    {
        for(int c = 32; c > 0; c >>= 1)
        {
            dst_block[threadIdx.x] = ::hypot(
                    dst_block[threadIdx.x], dst_block[threadIdx.x+c]);
        }
    }
    if(threadIdx.x == 0)
    {
        Y scaled_src2_val = beta * static_cast<Y>(src2.at(blockIdx.x)[0]);
        T *output_dst = dst.at(blockIdx.x);
        if(beta == 0.0)
        {
            output_dst[0] = static_cast<Y>(dst_block[0]);
        }
        else
        {
            output_dst[0] = ::hypot(scaled_src2_val, dst_block[0]);
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src1_chunks, Index nsrc1,
        Scalar beta, const T* const* src2_chunks, Index nsrc2,
        T* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(nsrc1 <= 0 || nsrc2 <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkConst<T> src1 = chunk_const<T>(src1_chunks, nsrc1, elems_src1, cls,
        "nntile_norm_slice");
    ChunkConst<T> src2 = chunk_const<T>(src2_chunks, nsrc2, elems_src2, cls,
        "nntile_norm_slice");
    ChunkMut<T> dst = chunk_mut<T>(dst_chunks, ndst, elems_dst, cls,
        "nntile_norm_slice");
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(n);
        (chunked_cuda_kernel_m1<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(
                n, k, alpha, src1, beta, src2, dst);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k,
                alpha, src1, beta, src2, dst);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, Scalar beta, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, Scalar beta, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, Scalar beta, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src1, Scalar beta, const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const fp32_t* const* src2_chunks, Index nsrc2,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp64_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const fp64_t* const* src2_chunks, Index nsrc2,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const bf16_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const bf16_t* const* src2_chunks, Index nsrc2,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp16_t* const* src1_chunks, Index nsrc1,
        Scalar beta, const fp16_t* const* src2_chunks, Index nsrc2,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::norm_slice
