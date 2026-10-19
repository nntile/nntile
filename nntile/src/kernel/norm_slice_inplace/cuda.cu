/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/norm_slice_inplace/cuda.cu
 * Euclidean norms of fibers into a slice of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/norm_slice_inplace/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::norm_slice_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index mk, Scalar alpha_,
        const T *src, Scalar beta_, T *dst)
//! Euclidean norms over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-n array, that
 *      accumulates norms along middle axis.
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    const Y beta{beta_};
    const Y alpha{alpha_};
    constexpr Y zero{0.};
    if(i0 < m and i1 < n)
    {
        // Pointer to a corresponding fiber of the source array src
        const T *src_fiber = src + i1*mk + i0;
        // Init sum over the fiber
        Y sum = zero;
        // Cycle over fiber elements and accumulate the sum
        for(Index i2 = i2_start; i2 < k; i2 += i2_step)
        {
            sum = ::hypot(sum, Y{src_fiber[i2*m]});
        }
        volatile __shared__ Y block_max[64];
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_max[threadIdx.x+blockDim.x*threadIdx.y] = sum;
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        while(block_max[threadIdx.x+blockDim.x*threadIdx.y] < sum)
        {
            block_max[threadIdx.x+blockDim.x*threadIdx.y] = sum;
        }
        __syncthreads();
        if(block_max[threadIdx.x+blockDim.x*threadIdx.y] > 0)
        {
            sum /= block_max[threadIdx.x+blockDim.x*threadIdx.y];
            atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y],
                    sum*sum);
        }
        __syncthreads();
        // Update output value
        if(i2_start == 0)
        {
            // Output value
            T &result = dst[i1*m+i0];
            if(block_max[threadIdx.x+blockDim.x*threadIdx.y] > 0)
            {
                sum = block_max[threadIdx.x+blockDim.x*threadIdx.y];
                sum *= ::sqrt(block_sum[threadIdx.x+blockDim.x*threadIdx.y]);
                if(beta == zero)
                {
                    result = T{::fabs(alpha) * sum};
                }
                else
                {
                    result = T{::hypot(beta*Y{result}, alpha*sum)};
                }
            }
            // All-zero fiber: write the beta-combined value exactly like
            // the CPU kernel and the m==1 CUDA kernel of this family do
            // (0 for beta==0, fabs(beta*dst_old) otherwise). The old code
            // skipped the write entirely, leaving dst stale - divergent
            // from the CPU kernel on legal zero inputs (padding, masks)
            // and fatal for the piecewise hypot chain of the chunked
            // driver, whose continuation pieces anchor on the first
            // piece's output.
            else
            {
                if(beta == zero)
                {
                    result = T{zero};
                }
                else
                {
                    result = T{::fabs(beta*Y{result})};
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Euclidean norms over fibers along middle axis into a slice of a tensor
/*! For a provided 1-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in 1-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[0,j] = hypot(beta*dst[0,j], alpha*norm(src[0,:,j]))
 *
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha_: Scaling factor for src
 * @param[in] src: Input contiguous 1-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous 1-by-n array, that
 *      accumulates norms along middle axis.
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
            dst_val = ::hypot(dst_val, val);
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index src_l = threadIdx.x + src_l_block_end;
    const T *src_fiber = src + blockIdx.x*k;
    for(Index c = src_l; c < k; c += BLOCK_ROW_STEP)
    {
        Y val = static_cast<Y>(src_fiber[c]);
        dst_val = ::hypot(dst_val, val);
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
        if(beta == 0.0)
        {
            dst[blockIdx.x] = static_cast<T>(static_cast<Y>(dst_block[0]));
        }
        else
        {
            dst_val = beta * static_cast<Y>(dst[blockIdx.x]);
            dst[blockIdx.x] = static_cast<T>(::hypot(dst_val, dst_block[0]));
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src, Scalar beta, T *dst)
    noexcept
//! Euclidean norms over fibers along middle axis into a slice of a tensor
/*! For a provided m-by-k-by-n input array src compute norms of fibers
 * along second axis with k elements, resulting in m-by-n output array-slice
 * dst.
 * Mnemonically, the following operations are performed:
 *      dst[i,j] = hypot(beta*dst[i,j], alpha*norm(src[i,:,j]))
 *
 * @param[in] m: Size of the first mode of src and dst arrays
 * @param[in] n: Size of the last mode of src and dst arrays
 * @param[in] k: Size of the middle mode of src array
 * @param[in] alpha: Scaling factor for src
 * @param[in] src_: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst_: Input and output contiguous m-by-n array, that
 *      accumulates norms along middle axis.
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
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src, beta, dst);
    }
}

//! Chunk descriptors for the one-launch chunked norm_slice_inplace.
template<typename T>
struct NormSliceChunkTables
{
    const T* const* src; //!< device-ring view of the src pointer array
    T* const* dst; //!< device-ring view of the dst pointer array
    Index nsrc;
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index m;
    Index n;
    Index k;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long nsl_chunk_of(long f, Index elems,
    int shift)
{
    return f >> shift;
}

//! Chunk-aware m==1 kernel: the SAME geometry and accumulation order as
//! cuda_kernel_m1 above, reads resolved through the src chunk table and
//! the write through the dst table. Deterministic (no atomics), so bitwise
//! cuda()->m1.
template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void chunked_m1_kernel(NormSliceChunkTables<T> t)
{
    Index const elems_src = t.elems_src;
    Index const elems_dst = t.elems_dst;
    int const shift_src = t.shift_src;
    int const shift_dst = t.shift_dst;
    Index const j = blockIdx.x;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    const Y beta{t.beta};
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    volatile __shared__ Y dst_block[BLOCK_ROW_STEP];
    Y dst_val = 0.0;
    long const base = long(j) * long(t.k);
    long const step = BLOCK_ROW_STEP;
    // Collapse the two original loops into one affine walk (offsets
    // tid + step*i < k) and rebase the chunk pointer only on a boundary
    // crossing (see sum_slice's chunked_m1_kernel).
    long f = base + long(threadIdx.x);
    long const end = base + long(t.k);
    if(shift_src >= 0)
    {
        if(threadIdx.x < t.k)
        {
            Index ci = Index(f >> shift_src);
            long off = f - (long(ci) << shift_src);
            T const *p = t.src[ci];
            for(; f < end; f += step)
            {
                dst_val = ::hypot(dst_val, Y{p[off]});
                if(f + step >= end) { break; }
                off += step;
                if(off >= elems_src)
                {
                    do { off -= elems_src; ++ci; } while(off >= elems_src);
                    p = t.src[ci];
                }
            }
        }
    }
    else
    {
        for(; f < end; f += step)
        {
            long const ci = nsl_chunk_of(f, elems_src, shift_src);
            dst_val = ::hypot(dst_val, Y{t.src[ci][f - ci * long(elems_src)]});
        }
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
        long const g = long(j);
        long const cd = nsl_chunk_of(g, elems_dst, shift_dst);
        T *dp = t.dst[cd] + (g - cd * long(elems_dst));
        if(beta == 0.0)
        {
            dp[0] = T{Y(dst_block[0])};
        }
        else
        {
            Y v = beta * Y{dp[0]};
            dp[0] = T{::hypot(v, dst_block[0])};
        }
    }
}

//! Chunk-aware general kernel (m>1): same geometry and accumulation order
//! as cuda_kernel above (blocked per-fiber hypot, shared max rescale,
//! shared atomicAdd), addresses resolved through the chunk tables.
template<typename T>
static __global__
void chunked_general_kernel(NormSliceChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    const Y beta{t.beta};
    const Y alpha{t.alpha};
    constexpr Y zero{0.};
    if(i0 < t.m and i1 < t.n)
    {
        long const mk = long(t.m) * long(t.k);
        Y sum = zero;
        for(Index i2 = i2_start; i2 < t.k; i2 += i2_step)
        {
            long const f = long(i1)*mk + long(i0) + long(i2)*long(t.m);
            long const ci = nsl_chunk_of(f, t.elems_src, t.shift_src);
            sum = ::hypot(sum, Y{t.src[ci][f - ci * long(t.elems_src)]});
        }
        volatile __shared__ Y block_max[64];
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_max[threadIdx.x+blockDim.x*threadIdx.y] = sum;
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        while(block_max[threadIdx.x+blockDim.x*threadIdx.y] < sum)
        {
            block_max[threadIdx.x+blockDim.x*threadIdx.y] = sum;
        }
        __syncthreads();
        if(block_max[threadIdx.x+blockDim.x*threadIdx.y] > 0)
        {
            sum /= block_max[threadIdx.x+blockDim.x*threadIdx.y];
            atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y],
                    sum*sum);
        }
        __syncthreads();
        if(i2_start == 0)
        {
            long const g = long(i1)*long(t.m) + long(i0);
            long const cd = nsl_chunk_of(g, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + (g - cd * long(t.elems_dst));
            if(block_max[threadIdx.x+blockDim.x*threadIdx.y] > 0)
            {
                sum = block_max[threadIdx.x+blockDim.x*threadIdx.y];
                sum *= ::sqrt(block_sum[threadIdx.x+blockDim.x*threadIdx.y]);
                if(beta == zero)
                {
                    dp[0] = T{::fabs(alpha) * sum};
                }
                else
                {
                    dp[0] = T{::hypot(beta*Y{dp[0]}, alpha*sum)};
                }
            }
            else
            {
                if(beta == zero)
                {
                    dp[0] = T{zero};
                }
                else
                {
                    dp[0] = T{::fabs(beta*Y{dp[0]})};
                }
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 or n <= 0 or k <= 0 or nsrc <= 0 or ndst <= 0)
    {
        return;
    }
    NormSliceChunkTables<T> t;
    t.src = src_chunks;
    t.dst = dst_chunks;
    t.nsrc = nsrc;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_norm_slice_inplace");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_norm_slice_inplace");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    t.beta = beta;
    // Mirror cuda()'s branch selection and geometry.
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(n);
        (chunked_m1_kernel<T, 1024, 4>)<<<blocks, threads, 0,
            stream>>>(t);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (chunked_general_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
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

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_t* const* src_chunks, Index nsrc, Scalar beta,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp64_t* const* src_chunks, Index nsrc, Scalar beta,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const bf16_t* const* src_chunks, Index nsrc, Scalar beta,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp16_t* const* src_chunks, Index nsrc, Scalar beta,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;






} // namespace nntile::kernel::norm_slice_inplace
