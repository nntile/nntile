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

#include <cstdio>
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

//! Chunk descriptors for the one-launch chunked sum_slice (see cuda.hh).
template<typename T>
struct SumSliceChunkTables
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
static __device__ __forceinline__ long ss_chunk_of(long f, Index elems,
    int shift)
{
    return f >> shift;
}

//! Per-thread partial sum of one output column of the chunked m==1
//! kernel: the row lies inside one chunk of the source (every
//! single-chunk operand and every interior row of a class-chunked one).
//! Resolve the chunk home ONCE into a virtual contiguous base
//! q = chunk_base - chunk*elems (the element at flat f is q[f]) and run
//! cuda_kernel_m1's two loops VERBATIM through it -- the piece-4
//! "single-run fast path" of multiply_slice/sumprod_slice. Same
//! elements, same per-thread ascending order, same `dst_val += val`
//! statements as the classic kernel, hence a bitwise-identical partial.
template<typename T, int BLOCK_ROW, int LOOP>
static __device__ __forceinline__
typename T::repr_t chunked_m1_single_run(
        SumSliceChunkTables<T> const &t, Index j, long c1)
{
    using Y = typename T::repr_t;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    long const base = long(j) * long(t.k);
    T const *q1 = t.src[c1] - c1 * long(t.elems_src);
    Y dst_val = 0.0;
    Index const src_l_block_end = (t.k/BLOCK_ROW) * BLOCK_ROW;
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        const T *src_fiber = q1 + src_l + base;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y val = static_cast<Y>(src_fiber[c]);
            dst_val += val;
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index const src_l = threadIdx.x + src_l_block_end;
    const T *src_fiber = q1 + base;
    for(Index c = src_l; c < t.k; c += BLOCK_ROW_STEP)
    {
        Y val = static_cast<Y>(src_fiber[c]);
        dst_val += val;
    }
    return dst_val;
}

//! Cold path of the same partial: a straddling row (crosses a chunk
//! boundary of the source). One affine walk over the offsets
//! tid + step*i < k with the chunk pointer rebased only on a boundary
//! crossing (sumprod_slice's chunked_m1_straddle); the walk visits the
//! same elements in the same ascending order as the classic loops, so
//! the partial is again bitwise-identical.
template<typename T, int BLOCK_ROW, int LOOP>
static __device__ __forceinline__
typename T::repr_t chunked_m1_straddle(
        SumSliceChunkTables<T> const &t, Index j)
{
    using Y = typename T::repr_t;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    Index const e1 = t.elems_src;
    int const shf1 = t.shift_src;
    Y dst_val = 0.0;
    long const base = long(j) * long(t.k);
    long const step = BLOCK_ROW_STEP;
    long f = base + long(threadIdx.x);
    long const end = base + long(t.k);
    if(threadIdx.x < t.k)
    {
        Index cl1 = Index(f >> shf1);
        long off1 = f - (long(cl1) << shf1);
        T const *p1 = t.src[cl1];
        for(; f < end; f += step)
        {
            dst_val += Y{p1[off1]};
            if(f + step >= end) { break; }
            off1 += step;
            if(off1 >= e1)
            {
                do { off1 -= e1; ++cl1; } while(off1 >= e1);
                p1 = t.src[cl1];
            }
        }
    }
    return dst_val;
}

//! Chunk-aware m==1 kernel: the SAME per-output accumulation order as
//! cuda_kernel_m1 above, reads resolved through the src chunk table and
//! the write through the dst table. Deterministic (no atomics), so
//! bitwise cuda()->m1. Each 256-thread block FUSES two consecutive
//! output columns (jA = 2*blockIdx.x, jB = jA+1), the critic-approved
//! sumprod_slice shape: both per-thread partials accumulate first
//! (independent loads interleave, which hides the chunk-table load
//! latency), then ONE reduction pass serves both columns with the
//! classic add tree per column (write, barrier, +128-step, barrier,
//! +64-step, barrier, then the 32..1 warp tree) -- the tree shape,
//! element set and add order per output are exactly cuda_kernel_m1's,
//! so the results are bitwise-identical while the per-block start cost
//! and barrier count are halved per column. The in-warp tail of the
//! tree runs in registers through __shfl_down_sync (x = v[i]+v[i+32]
//! then offsets 16,8,4,2,1), which performs the identical association
//! as the classic volatile shared-memory loop.
template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void chunked_m1_kernel(SumSliceChunkTables<T> t)
{
    Index const e1 = t.elems_src, ed = t.elems_dst;
    int const shf1 = t.shift_src, shfd = t.shift_dst;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    const Y beta{t.beta};
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    __shared__ Y dstb[2][BLOCK_ROW_STEP];
    Index const jA = blockIdx.x * 2;
    Index const jB = jA + 1;
    bool const hasB = jB < t.n;
    // Both columns' partials, then the row's chunk homes (shift/mask, no
    // divisors); single-chunk-row fast path or the straddling walk.
    long const baseA = long(jA) * long(t.k);
    long const c1A = baseA >> shf1;
    long const off1A = baseA - c1A * long(e1);
    Y va = (off1A + long(t.k) <= long(e1))
        ? chunked_m1_single_run<T, BLOCK_ROW, LOOP>(t, jA, c1A)
        : chunked_m1_straddle<T, BLOCK_ROW, LOOP>(t, jA);
    Y vb = 0.0;
    if(hasB)
    {
        long const baseB = long(jB) * long(t.k);
        long const c1B = baseB >> shf1;
        long const off1B = baseB - c1B * long(e1);
        vb = (off1B + long(t.k) <= long(e1))
            ? chunked_m1_single_run<T, BLOCK_ROW, LOOP>(t, jB, c1B)
            : chunked_m1_straddle<T, BLOCK_ROW, LOOP>(t, jB);
    }
    // The classic reduction, one fused pass for both columns.
    dstb[0][threadIdx.x] = alpha * va;
    if(hasB)
    {
        dstb[1][threadIdx.x] = alpha * vb;
    }
    __syncthreads();
    for(int c = BLOCK_ROW_STEP>>1; c > 32; c >>= 1)
    {
        if(threadIdx.x < c)
        {
            dstb[0][threadIdx.x] += dstb[0][threadIdx.x+c];
            if(hasB)
            {
                dstb[1][threadIdx.x] += dstb[1][threadIdx.x+c];
            }
        }
        __syncthreads();
    }
    if(threadIdx.x < 32)
    {
        Y x = dstb[0][threadIdx.x] + dstb[0][threadIdx.x+32];
        #pragma unroll
        for(int c = 16; c > 0; c >>= 1)
        {
            x += __shfl_down_sync(0xffffffffu, x, c);
        }
        if(threadIdx.x == 0)
        {
            long const g = long(jA);
            long const cd = ss_chunk_of(g, ed, shfd);
            T *dp = t.dst[cd] + (g - cd * long(ed));
            if(beta == 0.0)
            {
                dp[0] = T{Y(x)};
            }
            else
            {
                Y v = beta * Y{dp[0]};
                dp[0] = T{v + x};
            }
        }
    }
    if(threadIdx.x < 32 and hasB)
    {
        Y x = dstb[1][threadIdx.x] + dstb[1][threadIdx.x+32];
        #pragma unroll
        for(int c = 16; c > 0; c >>= 1)
        {
            x += __shfl_down_sync(0xffffffffu, x, c);
        }
        if(threadIdx.x == 0)
        {
            long const g = long(jB);
            long const cd = ss_chunk_of(g, ed, shfd);
            T *dp = t.dst[cd] + (g - cd * long(ed));
            if(beta == 0.0)
            {
                dp[0] = T{Y(x)};
            }
            else
            {
                Y v = beta * Y{dp[0]};
                dp[0] = T{v + x};
            }
        }
    }
}

//! Chunk-aware general kernel (m>1): the SAME geometry and accumulation
//! order as cuda_kernel above (per-thread strided fiber walk with i2 outer,
//! shared-memory atomicAdd reduction), with reads/writes resolved through
//! the chunk tables. Equal to cuda() up to the atomic ordering floor.
template<typename T>
static __global__
void chunked_general_kernel(SumSliceChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    Y alpha{t.alpha};
    Y beta{t.beta};
    constexpr Y zero{0.0};
    if(i0 < t.m and i1 < t.n)
    {
        Y sum{zero};
        long const mk = long(t.m) * long(t.k);
        for(Index i2 = i2_start; i2 < t.k; i2 += i2_step)
        {
            long const f = long(i1)*mk + long(i0) + long(i2)*long(t.m);
            long const ci = ss_chunk_of(f, t.elems_src, t.shift_src);
            sum += Y{t.src[ci][f - ci * long(t.elems_src)]};
        }
        __shared__ Y block_sum[64];
        if(i2_start == 0)
        {
            block_sum[threadIdx.x+blockDim.x*threadIdx.y] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x+blockDim.x*threadIdx.y], sum);
        __syncthreads();
        if(i2_start == 0)
        {
            long const g = long(i1)*long(t.m) + long(i0);
            long const cd = ss_chunk_of(g, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + (g - cd * long(t.elems_dst));
            sum = block_sum[threadIdx.x+blockDim.x*threadIdx.y];
            if(beta == zero)
            {
                dp[0] = T{alpha * sum};
            }
            else
            {
                dp[0] = T{beta * Y{dp[0]} + alpha * sum};
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
    SumSliceChunkTables<T> t;
    t.src = src_chunks;
    t.dst = dst_chunks;
    t.nsrc = nsrc;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_sum_slice");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_sum_slice");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    t.beta = beta;
    // Mirror cuda()'s branch selection and geometry; the m1 kernel fuses
    // two columns per block (see chunked_m1_kernel).
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks((n + 1) / 2);
        (chunked_m1_kernel<T, 1024, 4>)<<<blocks, threads, 0,
            stream>>>(t);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (chunked_general_kernel<T>)<<<blocks, threads, 0,
            stream>>>(t);
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






} // namespace nntile::kernel::sum_slice
