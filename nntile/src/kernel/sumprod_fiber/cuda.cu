/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sumprod_fiber/cuda.cu
 * Sums over slices into a fiber of a product of buffers on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/sumprod_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sumprod_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, Scalar beta_, T *dst)
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha_: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    constexpr Y zero{0.0};
    // Init sum of product of the slices
    Y sum = zero;
    if(i2 < k)
    {
        // Cycle over column of src1 and src2
        for(Index i1 = i1_start; i1 < n; i1 += i1_step)
        {
            // Get corresponding fibers of both sources
            const T *src1_fiber = src1 + (i1*k+i2)*m;
            const T *src2_fiber = src2 + (i1*k+i2)*m;
            // Cycle over fibers of inputs
            for(Index i0 = i0_start; i0 < m; i0 += i0_step)
            {
                // Update sum
                sum += Y{src1_fiber[i0]} * Y{src2_fiber[i0]};
            }
        }
    }
    __shared__ Y block_sum[2];
    if(i1_start == 0 and i0_start == 0)
    {
        block_sum[threadIdx.x] = zero;
    }
    __syncthreads();
    atomicAdd(&block_sum[threadIdx.x], sum);
    __syncthreads();
    // Update output value
    if(i1_start == 0 and i0_start == 0 and i2 < k)
    {
        // Output value
        T &result = dst[i2];
        if(beta == zero)
        {
            result = T{alpha * Y{block_sum[threadIdx.x]}};
        }
        else
        {
            result = T{beta * Y{result} + alpha * block_sum[threadIdx.x]};
        }
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, Scalar beta_, T *dst)
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided 1-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_j(src1[0,l,j] * src2[0,l,j])
 *
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha_: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous 1-by-k-by-n array
 * @param[in] src2: Input contiguous 1-by-k-by-n array
 * @param[in] beta_: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    Index src_block_j_end = (n/BLOCK_COL) * BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    __shared__ Y dst_block[BLOCK_ROW][BLOCK_COL_STEP];
    Y dst_val = 0.0;
    Index src_l = threadIdx.x % BLOCK_ROW;
    Index src_j = threadIdx.x / BLOCK_ROW;
    Index src_offset = blockIdx.x*BLOCK_ROW + src_l + src_j*k;
    // Pointer to a corresponding fiber of the input arrays
    if(src_l+blockIdx.x*BLOCK_ROW < k)
    {
        for(Index src_block_j = 0; src_block_j < src_block_j_end;
                src_block_j += BLOCK_COL)
        {
            const T *src1_fiber = src1 + src_offset + src_block_j*k;
            const T *src2_fiber = src2 + src_offset + src_block_j*k;
            for(int c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                // Threads beyond the last column must not read the
                // source arrays: those elements belong to other data.
                if(src_block_j + src_j + c < n)
                {
                    Y val1 = static_cast<Y>(src1_fiber[c*k]);
                    Y val2 = static_cast<Y>(src2_fiber[c*k]);
                    dst_val += val1 * val2;
                }
            }
        }
        // Pointer to a corresponding fiber of the input arrays
        const T *src1_fiber = src1 + src_offset + src_block_j_end*k;
        const T *src2_fiber = src2 + src_offset + src_block_j_end*k;
        for(Index c = 0; c < n-src_block_j_end; c += BLOCK_COL_STEP)
        {
            // Same bound as above: src_j offsets this thread's column.
            if(src_block_j_end + src_j + c < n)
            {
                Y val1 = static_cast<Y>(src1_fiber[c*k]);
                Y val2 = static_cast<Y>(src2_fiber[c*k]);
                dst_val += val1 * val2;
            }
        }
    }
    // Put calculated value into shared memory
    dst_block[src_l][src_j] = alpha * dst_val;
    // Inter-warp reduction
    for(int c = BLOCK_COL_STEP>>1; c > 0; c >>= 1)
    {
        __syncthreads();
        if(src_j < c)
        {
            dst_block[src_l][src_j] += dst_block[src_l][src_j+c];
        }
    }
    // Write output
    if(src_j == 0 and src_l+blockIdx.x*BLOCK_ROW < k)
    {
        if(beta == 0.0)
        {
            dst[blockIdx.x*BLOCK_ROW+src_l] = static_cast<T>(
                    static_cast<Y>(dst_block[src_l][0]));
        }
        else
        {
            dst_val = beta * static_cast<Y>(dst[blockIdx.x*BLOCK_ROW+src_l]);
            dst[blockIdx.x*BLOCK_ROW+src_l] = static_cast<T>(
                    dst_val + dst_block[src_l][0]);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, Scalar beta, T *dst)
    noexcept
//! Sums over slices into a fiber of a product of two tensors on CPU
/*! For two provided m-by-k-by-n input arrays src1 and src2 compute sums of
 * per-element product of corresponding slices along the first and the third
 * axes with m and n elements respectively, resulting in output vector dst with
 * k elements.
 * Mnemonically, the following operations are performed:
 *      dst[l] = beta*dst[l] + alpha*sum_ij(src1[i,l,j] * src2[i,l,j])
 *
 * @param[in] m: Size of the first mode of src1 and src2 tensors
 * @param[in] n: Size of the last mode of src1 and src2 tensors
 * @param[in] k: Size of the middle mode of src1 and src2 tensors and of the
 *      only mode of dst tensor
 * @param[in] alpha: Scaling factor for src1*src2
 * @param[in] src1: Input contiguous m-by-k-by-n array
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that
 *      accumulates sums along the first and the last axes of per-element
 *      products of src1 and src2.
 * */
{
    // Both source and destination are Fortran-contiguous
    // Separate case for m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks((k+31)/32);
        (cuda_kernel_m1<T, 32, 8, 1>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, src2, beta, dst);
    }
    else
    {
        // Cap the block at 1024 threads: 2 x min(m,32) x min(n,32) alone
        // exceeds the limit whenever min(m,32)*min(n,32) > 512 (a launch
        // that always failed with invalid configuration argument).
        // Mapping-stability caveat (family-2 follow-up B, corrected): the
        // thread-to-element mapping is preserved ONLY while
        // 2*ty*min(n,32) <= 512, where tz = min(n,32) still assigns one
        // thread per (i0, i2) fiber position. In the (512, 1024]-thread
        // band - 512 < 2*ty*min(n,32) <= 1024, i.e. shapes whose original
        // 2 x ty x min(n,32) block was legal but larger than 512 threads -
        // tz = max(1, 512/(2*ty)) < min(n,32) REGROUPS the partial sums
        // (each i2 accumulator folds a different thread's strided partial
        // before the shared-memory atomicAdd), so such shapes change their
        // summation grouping versus the pre-cap kernel. Correctness is
        // unaffected: the kernel's reduction is atomicAdd-based and its
        // order is unspecified anyway, so any grouping is a legal answer.
        int const ty = std::min(int(m), 32);
        int const tz = std::min(int(n), std::max(1, 512 / (2 * ty)));
        dim3 threads(2, ty, tz);
        dim3 blocks((k+threads.x-1)/threads.x, 1, 1);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src1,
                src2, beta, dst);
    }
}

//! Chunk descriptors for the one-launch chunked sumprod_fiber (see cuda.hh).
template<typename T>
struct SumProdFiberChunkTables
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
static __device__ __forceinline__ long spf_chunk_of(long f, Index elems,
    int shift)
{
    return f >> shift;
}

template<typename T>
static __device__ __forceinline__ T spf_at(const T* const* table, long f,
    Index elems, int shift)
{
    long const c = spf_chunk_of(f, elems, shift);
    return table[c][f - c * long(elems)];
}

//! Chunk-aware m==1 kernel: the SAME geometry ((src_l, src_j) work
//! assignment, BLOCK_COL column grouping, shared tree reduction) and the
//! SAME accumulation order as cuda_kernel_m1 above, reads resolved through
//! the two src chunk tables and the write through the dst table.
//! Deterministic (no atomics), so bitwise cuda()->m1.
template<typename T, int BLOCK_ROW, int BLOCK_COL, int LOOP>
static __global__
void chunked_m1_kernel(SumProdFiberChunkTables<T> t)
{
    Index const e1 = t.elems_src1, e2 = t.elems_src2, ed = t.elems_dst;
    int const shf1 = t.shift_src1, shf2 = t.shift_src2, shfd = t.shift_dst;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    const Y beta{t.beta};
    constexpr int BLOCK_COL_STEP = BLOCK_COL / LOOP;
    __shared__ Y dst_block[BLOCK_ROW][BLOCK_COL_STEP];
    Y dst_val = 0.0;
    Index const src_l = threadIdx.x % BLOCK_ROW;
    Index const src_j = threadIdx.x / BLOCK_ROW;
    Index const base_l = blockIdx.x*BLOCK_ROW + src_l;
    if(base_l < t.k)
    {
        // Run-structured in-chunk walk (piece 4). Each thread reads the
        // flat offsets f0 + i*step (step = BLOCK_COL*k), i = 0, 1, ...,
        // while f < k + src_j... i.e. exactly the column sequence of
        // cuda_kernel_m1 above, in the same order. The span-start divmod
        // runs ONCE per thread; inside a run the element at flat offset f
        // is addressed as q[f] through a virtual contiguous base
        // q = chunk_base - chunk*elems, so the hot loop is branch-free and
        // nvcc pipelines the loads exactly like the classic kernel's loop
        // (the earlier per-boundary branch inside the walk inhibited
        // unrolling and starved memory-level parallelism: the GPT-2
        // (m,n,k)=(1,1024,768) shape measured 2.9x classic even on the
        // power-of-two shift path). A rebase (one divmod per chunk
        // boundary, i.e. per run that straddles) restores q when the walk
        // crosses into the next chunk. Same values in the same order, so
        // the result is unchanged bitwise.
        long const step = long(BLOCK_COL) * long(t.k);
        long const f0 = long(base_l) + long(src_j) * long(t.k);
        long const fend = long(base_l) + long(t.n) * long(t.k);
        if(src_j < t.n)
        {
            // Chunk residency of the walk's first and last read (two
            // divmods at span granularity): if the whole per-thread
            // walk stays inside one chunk of both sources, run the
            // plain affine loop through the virtual contiguous bases --
            // the exact loop shape of cuda_kernel_m1, so nvcc pipelines
            // the loads identically (this covers every 1-chunk operand
            // and the interior spans of multi-chunk ones). Spans that
            // straddle a boundary take the run walk below.
            long const flast = fend - 1 - ((fend - 1 - f0) % step);
            // (rule 7) span-start chunk residency by shift/mask on the
            // per-operand shifts (class shift or the 62 sentinel), never
            // a runtime divisor.
            long const c1 = f0 >> shf1;
            long const c2 = f0 >> shf2;
            if(flast >> shf1 == c1 && flast >> shf2 == c2)
            {
                T const *q1 = t.src1[c1] - c1 * e1;
                T const *q2 = t.src2[c2] - c2 * e2;
                for(long f = f0; f < fend; f += step)
                {
                    dst_val += Y{q1[f]} * Y{q2[f]};
                }
            }
            else
            {
                long off1 = f0 - c1 * e1;
                T const *q1 = t.src1[c1] - c1 * e1;
                long off2 = f0 - c2 * e2;
                T const *q2 = t.src2[c2] - c2 * e2;
                long f = f0;
                while(f < fend)
                {
                    // Steps until either source's offset would leave its
                    // chunk (a read at offset o needs o < elems).
                    long const room1 = (e1 - off1 + step - 1) / step;
                    long const room2 = (e2 - off2 + step - 1) / step;
                    long const room = room1 < room2 ? room1 : room2;
                    long run_end = f + room * step;
                    if(run_end > fend)
                    {
                        run_end = fend;
                    }
                    for(; f < run_end; f += step)
                    {
                        dst_val += Y{q1[f]} * Y{q2[f]};
                        off1 += step;
                        off2 += step;
                    }
                    if(off1 >= e1)
                    {
                        long const c = f >> shf1;
                        off1 = f - c * e1;
                        q1 = t.src1[c] - c * e1;
                    }
                    if(off2 >= e2)
                    {
                        long const c = f >> shf2;
                        off2 = f - c * e2;
                        q2 = t.src2[c] - c * e2;
                    }
                }
            }
        }
    }
    dst_block[src_l][src_j] = alpha * dst_val;
    for(int c = BLOCK_COL_STEP>>1; c > 0; c >>= 1)
    {
        __syncthreads();
        if(src_j < c)
        {
            dst_block[src_l][src_j] += dst_block[src_l][src_j+c];
        }
    }
    if(src_j == 0 and base_l < t.k)
    {
        long const g = long(base_l);
        long const cd = spf_chunk_of(g, ed, shfd);
        T *dp = t.dst[cd] + (g - cd * long(ed));
        if(beta == 0.0)
        {
            dp[0] = T{Y(dst_block[src_l][0])};
        }
        else
        {
            Y v = beta * Y{dp[0]};
            dp[0] = T{v + dst_block[src_l][0]};
        }
    }
}

//! Chunk-aware general kernel (m>1): same geometry and accumulation order
//! as cuda_kernel above, addresses resolved through the chunk tables.
template<typename T>
static __global__
void chunked_general_kernel(SumProdFiberChunkTables<T> t)
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    const Y beta{t.beta};
    constexpr Y zero{0.0};
    Y sum = zero;
    if(i2 < t.k)
    {
        for(Index i1 = i1_start; i1 < t.n; i1 += i1_step)
        {
            long const plane = (long(i1)*long(t.k) + long(i2))
                * long(t.m);
            for(Index i0 = i0_start; i0 < t.m; i0 += i0_step)
            {
                long const f = plane + long(i0);
                Y val1 = Y{spf_at(t.src1, f, t.elems_src1, t.shift_src1)};
                Y val2 = Y{spf_at(t.src2, f, t.elems_src2, t.shift_src2)};
                sum += val1 * val2;
            }
        }
    }
    __shared__ Y block_sum[2];
    if(i1_start == 0 and i0_start == 0)
    {
        block_sum[threadIdx.x] = zero;
    }
    __syncthreads();
    atomicAdd(&block_sum[threadIdx.x], sum);
    __syncthreads();
    if(i1_start == 0 and i0_start == 0 and i2 < t.k)
    {
        long const g = long(i2);
        long const cd = spf_chunk_of(g, t.elems_dst, t.shift_dst);
        T *dp = t.dst[cd] + (g - cd * long(t.elems_dst));
        if(beta == zero)
        {
            dp[0] = T{alpha * Y{block_sum[threadIdx.x]}};
        }
        else
        {
            dp[0] = T{beta * Y{dp[0]} + alpha * Y{block_sum[threadIdx.x]}};
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
    SumProdFiberChunkTables<T> t;
    t.src1 = src1_chunks;
    t.src2 = src2_chunks;
    t.dst = dst_chunks;
    t.nsrc1 = nsrc1;
    t.nsrc2 = nsrc2;
    t.ndst = ndst;
    t.elems_src1 = elems_src1;
    t.elems_src2 = elems_src2;
    t.elems_dst = elems_dst;
    t.shift_src1 = chunk_shift_of<T>(t.nsrc1, elems_src1, "nntile_sumprod_fiber");
    t.shift_src2 = chunk_shift_of<T>(t.nsrc2, elems_src2, "nntile_sumprod_fiber");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_sumprod_fiber");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    t.beta = beta;
    // Mirror cuda()'s branch selection and geometry.
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks((k+31)/32);
        (chunked_m1_kernel<T, 32, 8, 1>)<<<blocks, threads, 0,
            stream>>>(t);
    }
    else
    {
        int const ty = std::min(int(m), 32);
        int const tz = std::min(int(n), std::max(1, 512 / (2 * ty)));
        dim3 threads(2, ty, tz);
        dim3 blocks((k+threads.x-1)/threads.x, 1, 1);
        (chunked_general_kernel<T>)<<<blocks, threads, 0,
            stream>>>(t);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src1, const fp16_t *src2, Scalar beta, fp16_t *dst)
    noexcept;

#define NNTILE_SUMPROD_FIBER_CHUNKED_INST(T)                                  \
template                                                                      \
void chunked<T>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha, \
        const T* const* src1_chunks, Index nsrc1,                             \
        const T* const* src2_chunks, Index nsrc2, Scalar beta,                \
        T* const* dst_chunks, Index ndst, Index elems_src1, Index elems_src2, \
        Index elems_dst) noexcept;

NNTILE_SUMPROD_FIBER_CHUNKED_INST(fp32_t)
NNTILE_SUMPROD_FIBER_CHUNKED_INST(fp64_t)
NNTILE_SUMPROD_FIBER_CHUNKED_INST(bf16_t)
NNTILE_SUMPROD_FIBER_CHUNKED_INST(fp16_t)
#undef NNTILE_SUMPROD_FIBER_CHUNKED_INST

} // namespace nntile::kernel::sumprod_fiber
