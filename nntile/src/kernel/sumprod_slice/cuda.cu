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

#include <cstdio>
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

template<typename T, int BLOCK_ROW, int LOOP>
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
        const T *src1_fiber = src1 + src_l + blockIdx.x*k;
        const T *src2_fiber = src2 + src_l + blockIdx.x*k;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y val1 = static_cast<Y>(src1_fiber[c]);
            Y val2 = static_cast<Y>(src2_fiber[c]);
            dst_val += val1 * val2;
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index src_l = threadIdx.x + src_l_block_end;
    const T *src1_fiber = src1 + blockIdx.x*k;
    const T *src2_fiber = src2 + blockIdx.x*k;
    for(Index c = src_l; c < k; c += BLOCK_ROW_STEP)
    {
        Y val1 = static_cast<Y>(src1_fiber[c]);
        Y val2 = static_cast<Y>(src2_fiber[c]);
        dst_val += val1 * val2;
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
        dim3 threads(256);
        dim3 blocks(n);
        (cuda_kernel_m1<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, src2, beta, dst);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8), 16);
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, m*k, alpha,
                src1, src2, beta, dst);
    }
}

//! Chunk descriptors for the one-launch chunked sumprod_slice (see cuda.hh).
template<typename T>
struct SumProdSliceChunkTables
{
    const T* const* src1; //!< device-ring view of the src1 pointer array
    const T* const* src2; //!< device-ring view of the src2 pointer array
    T* const* dst; //!< device-ring view of the dst pointer array
    Index nsrc1;
    Index nsrc2;
    Index ndst;
    Index elems_src1; //!< elements in a full src1 chunk
    Index elems_src2; //!< elements in a full src2 chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src1; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_src2; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index m;
    Index n;
    Index k;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long sps_chunk_of(long f, Index elems,
    int shift)
{
    return f >> shift;
}

template<typename T>
static __device__ __forceinline__ T sps_at(const T* const* table, long f,
    Index elems, int shift)
{
    long const c = sps_chunk_of(f, elems, shift);
    return table[c][f - c * long(elems)];
}

//! Per-thread partial sum of one output column of the chunked m==1
//! kernel: the row lies inside one chunk of BOTH sources (every
//! single-chunk operand and every interior row of a class-chunked one).
//! Resolve the two chunk homes ONCE into virtual contiguous bases
//! q = chunk_base - chunk*elems (the element at flat f is q[f]) and run
//! cuda_kernel_m1's two loops VERBATIM through them -- the piece-4
//! "single-run fast path" of multiply_slice. Same elements, same
//! per-thread ascending order, same val1*val2 statements as the classic
//! kernel, hence a bitwise-identical partial.
template<typename T, int BLOCK_ROW, int LOOP>
static __device__ __forceinline__
typename T::repr_t chunked_m1_single_run(
        SumProdSliceChunkTables<T> const &t, Index j, long c1, long c2)
{
    using Y = typename T::repr_t;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    long const base = long(j) * long(t.k);
    T const *q1 = t.src1[c1] - c1 * long(t.elems_src1);
    T const *q2 = t.src2[c2] - c2 * long(t.elems_src2);
    Y dst_val = 0.0;
    Index const src_l_block_end = (t.k/BLOCK_ROW) * BLOCK_ROW;
    for(Index src_l = threadIdx.x; src_l < src_l_block_end;
            src_l += BLOCK_ROW)
    {
        const T *src1_fiber = q1 + src_l + base;
        const T *src2_fiber = q2 + src_l + base;
        for(int c = 0; c < BLOCK_ROW; c += BLOCK_ROW_STEP)
        {
            Y val1 = static_cast<Y>(src1_fiber[c]);
            Y val2 = static_cast<Y>(src2_fiber[c]);
            dst_val += val1 * val2;
        }
    }
    // Pointer to a corresponding fiber of the input arrays
    Index const src_l = threadIdx.x + src_l_block_end;
    const T *src1_fiber = q1 + base;
    const T *src2_fiber = q2 + base;
    for(Index c = src_l; c < t.k; c += BLOCK_ROW_STEP)
    {
        Y val1 = static_cast<Y>(src1_fiber[c]);
        Y val2 = static_cast<Y>(src2_fiber[c]);
        dst_val += val1 * val2;
    }
    return dst_val;
}

//! Cold path of the same partial: a straddling row (crosses a chunk
//! boundary of at least one source). One affine walk over the offsets
//! tid + step*i < k with the chunk pointers rebased only on a boundary
//! crossing (see sum_slice's chunked_m1_kernel); the walk visits the
//! same elements in the same ascending order as the classic loops.
//! (Deliberately NOT __noinline__: in the shipped build an outlined
//! device call materializes the whole 120-byte table struct through a
//! local-memory stack frame, which costs 5-7x.)
template<typename T, int BLOCK_ROW, int LOOP>
static __device__ __forceinline__
typename T::repr_t chunked_m1_straddle(
        SumProdSliceChunkTables<T> const &t, Index j)
{
    using Y = typename T::repr_t;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    Index const e1 = t.elems_src1, e2 = t.elems_src2;
    int const shf1 = t.shift_src1, shf2 = t.shift_src2;
    Y dst_val = 0.0;
    long const base = long(j) * long(t.k);
    long const step = BLOCK_ROW_STEP;
    long f = base + long(threadIdx.x);
    long const end = base + long(t.k);
    if(threadIdx.x < t.k)
    {
        Index cl1 = Index(f >> shf1);
        Index cl2 = Index(f >> shf2);
        long off1 = f - (long(cl1) << shf1);
        long off2 = f - (long(cl2) << shf2);
        T const *p1 = t.src1[cl1];
        T const *p2 = t.src2[cl2];
        for(; f < end; f += step)
        {
            dst_val += Y{p1[off1]} * Y{p2[off2]};
            if(f + step >= end) { break; }
            off1 += step;
            off2 += step;
            if(off1 >= e1)
            {
                do { off1 -= e1; ++cl1; } while(off1 >= e1);
                p1 = t.src1[cl1];
            }
            if(off2 >= e2)
            {
                do { off2 -= e2; ++cl2; } while(off2 >= e2);
                p2 = t.src2[cl2];
            }
        }
    }
    return dst_val;
}

//! Chunk-aware m==1 kernel: the SAME per-output accumulation order as
//! cuda_kernel_m1 above, reads resolved through the two src chunk tables
//! and the write through the dst table. Deterministic (no atomics), so
//! bitwise cuda()->m1. Each 256-thread block FUSES two consecutive
//! output columns (jA = 2*blockIdx.x, jB = jA+1): both per-thread
//! partials accumulate first (independent loads interleave, which hides
//! the chunk-table load latency), then ONE reduction pass serves both
//! columns with the classic add tree per column (write, barrier,
//! +128-step, barrier, +64-step, barrier, then the 32..1 warp tree) --
//! the tree shape, element set and FMA order per output are exactly
//! cuda_kernel_m1's, so the results are bitwise-identical while the
//! per-block start cost and barrier count are halved per column.
//! The in-warp tail of the tree runs in registers through
//! __shfl_down_sync (x = v[i]+v[i+32] then offsets 16,8,4,2,1), which
//! performs the identical association as the classic volatile
//! shared-memory loop.
template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void chunked_m1_kernel(SumProdSliceChunkTables<T> t)
{
    Index const e1 = t.elems_src1, e2 = t.elems_src2, ed = t.elems_dst;
    int const shf1 = t.shift_src1, shf2 = t.shift_src2, shfd = t.shift_dst;
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
    long const c1A = baseA >> shf1, c2A = baseA >> shf2;
    long const off1A = baseA - c1A * long(e1);
    long const off2A = baseA - c2A * long(e2);
    Y va = (off1A + long(t.k) <= long(e1)
            and off2A + long(t.k) <= long(e2))
        ? chunked_m1_single_run<T, BLOCK_ROW, LOOP>(t, jA, c1A, c2A)
        : chunked_m1_straddle<T, BLOCK_ROW, LOOP>(t, jA);
    Y vb = 0.0;
    if(hasB)
    {
        long const baseB = long(jB) * long(t.k);
        long const c1B = baseB >> shf1, c2B = baseB >> shf2;
        long const off1B = baseB - c1B * long(e1);
        long const off2B = baseB - c2B * long(e2);
        vb = (off1B + long(t.k) <= long(e1)
                and off2B + long(t.k) <= long(e2))
            ? chunked_m1_single_run<T, BLOCK_ROW, LOOP>(t, jB, c1B, c2B)
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
            long const cd = sps_chunk_of(g, ed, shfd);
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
            long const cd = sps_chunk_of(g, ed, shfd);
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

//! Chunk-aware general kernel (m>1): same geometry and accumulation order
//! as cuda_kernel above, addresses resolved through the chunk tables.
template<typename T>
static __global__
void chunked_general_kernel(SumProdSliceChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y;
    Index i2_start = threadIdx.z, i2_step = blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero{0.};
    const Y alpha{t.alpha};
    const Y beta{t.beta};
    if(i0 < t.m and i1 < t.n)
    {
        Y sum = zero;
        long const mk = long(t.m) * long(t.k);
        for(Index i2 = i2_start; i2 < t.k; i2 += i2_step)
        {
            long const f = long(i1)*mk + long(i0) + long(i2)*long(t.m);
            Y v1 = Y{sps_at(t.src1, f, t.elems_src1, t.shift_src1)};
            Y v2 = Y{sps_at(t.src2, f, t.elems_src2, t.shift_src2)};
            sum += v1 * v2;
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
            long const cd = sps_chunk_of(g, t.elems_dst, t.shift_dst);
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
        const T* const* src1_chunks, Index nsrc1,
        const T* const* src2_chunks, Index nsrc2, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src1, Index elems_src2,
        Index elems_dst)
    noexcept
{
    if(m <= 0 or n <= 0 or k <= 0 or nsrc1 <= 0 or nsrc2 <= 0 or ndst <= 0)
    {
        return;
    }
    SumProdSliceChunkTables<T> t;
    t.src1 = src1_chunks;
    t.src2 = src2_chunks;
    t.dst = dst_chunks;
    t.nsrc1 = nsrc1;
    t.nsrc2 = nsrc2;
    t.ndst = ndst;
    t.elems_src1 = elems_src1;
    t.elems_src2 = elems_src2;
    t.elems_dst = elems_dst;
    t.shift_src1 = chunk_shift_of<T>(t.nsrc1, elems_src1, "nntile_sumprod_slice");
    t.shift_src2 = chunk_shift_of<T>(t.nsrc2, elems_src2, "nntile_sumprod_slice");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_sumprod_slice");
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
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8), 16);
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y, 1);
        (chunked_general_kernel<T>)<<<blocks, threads, 0,
            stream>>>(t);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, Scalar beta, fp32_t *sum_dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_fast_tf32_t *src1,
        const fp32_fast_tf32_t *src2, Scalar beta, fp32_fast_tf32_t *sum_dst)
    noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_fast_fp16_t *src1,
        const fp32_fast_fp16_t *src2, Scalar beta, fp32_fast_fp16_t *sum_dst)
    noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Scalar alpha, const fp32_fast_bf16_t *src1,
        const fp32_fast_bf16_t *src2, Scalar beta, fp32_fast_bf16_t *sum_dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, Scalar beta, fp64_t *sum_dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, Scalar beta, bf16_t *sum_dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src1, const fp16_t *src2, Scalar beta, fp16_t *sum_dst)
    noexcept;

#define NNTILE_SUMPROD_SLICE_CHUNKED_INST(T)                                  \
template                                                                      \
void chunked<T>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha, \
        const T* const* src1_chunks, Index nsrc1,                             \
        const T* const* src2_chunks, Index nsrc2, Scalar beta,                \
        T* const* dst_chunks, Index ndst, Index elems_src1, Index elems_src2, \
        Index elems_dst) noexcept;

NNTILE_SUMPROD_SLICE_CHUNKED_INST(fp32_t)
NNTILE_SUMPROD_SLICE_CHUNKED_INST(fp32_fast_tf32_t)
NNTILE_SUMPROD_SLICE_CHUNKED_INST(fp32_fast_fp16_t)
NNTILE_SUMPROD_SLICE_CHUNKED_INST(fp32_fast_bf16_t)
NNTILE_SUMPROD_SLICE_CHUNKED_INST(fp64_t)
NNTILE_SUMPROD_SLICE_CHUNKED_INST(bf16_t)
NNTILE_SUMPROD_SLICE_CHUNKED_INST(fp16_t)
#undef NNTILE_SUMPROD_SLICE_CHUNKED_INST

} // namespace nntile::kernel::sumprod_slice
