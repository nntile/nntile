/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/multiply_fiber/cuda.cu
 * Per-element multiplication of a tensor by a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/multiply_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::multiply_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Scalar alpha_,
        const T * __restrict__ src1, const T * __restrict__ src2,
        T * __restrict__ dst)
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i0 < m and i1 < n and i2 < k)
    {
        const Y src1_val = alpha * Y{src1[i2]};
        // Input fiber to be used
        const T *src2_fiber = src2 + (i1*k+i2)*m;
        // Output fiber to be updated
        T *dst_fiber = dst + (i1*k+i2)*m;
        // Update output value
        dst_fiber[i0] = T{src1_val * Y{src2_fiber[i0]}};
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, Scalar alpha_, const T *src1,
        const T *src2, T *dst)
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[0,l,j] = alpha * src1[l] * src2[0,l,j]
 *
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha_: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous 1-by-k-by-n array
 * @param[out] dst: Output contiguous 1-by-k-by-n array
 * */
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    __shared__ Y src1_block[BLOCK_ROW];
    __shared__ T src2_block[BLOCK_ROW][BLOCK_COL+1];
    Index src1_l = dst_block_l*BLOCK_ROW + threadIdx.x;
    if(src1_l < k and threadIdx.x < BLOCK_ROW)
    {
        src1_block[threadIdx.x] = alpha * static_cast<Y>(src1[src1_l]);
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        const T *src2_fiber = src2 + global_dst_l + global_dst_j*k;
        T *dst_fiber = dst + global_dst_l + global_dst_j*k;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                src2_block[dst_l][c] = src2_fiber[c*k];
            }
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(src1_block[dst_l] *
                        static_cast<Y>(src2_block[dst_l][c]));
            }
        }
        else
        {
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                src2_block[dst_l][c] = src2_fiber[c*k];
            }
            for(Index c = 0; c < n-global_dst_j; c += BLOCK_COL_STEP)
            {
                dst_fiber[c*k] = static_cast<T>(src1_block[dst_l] *
                        static_cast<Y>(src2_block[dst_l][c]));
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const T *src1, const T *src2, T *dst)
    noexcept
//! Per-element product of a tensor and a broadcasted fiber on CPU
//! Per-element product of a tensor and a broadcasted fiber on CPU
/*! Performs the following operations:
 *      dst[i,l,j] = alpha * src1[l] * src2[i,l,j]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensor
 * @param[in] alpha: Scalar factor
 * @param[in] src1: Input contiguous vector with k elements
 * @param[in] src2: Input contiguous m-by-k-by-n array
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    // Custom version of code for special m==1 case
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0, stream>>>(n, k,
                alpha, src1, src2, dst);
    }
    // Generic case
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, alpha, src1,
                src2, dst);
    }
}

//! Chunk descriptors for the one-launch chunked multiply_fiber (see cuda.hh).
template<typename T>
struct ChunkTables
{
    const T *const* src1; //!< device-ring view of the src1 pointer array
    Index ns1;
    const T *const* src2; //!< device-ring view of the src2 pointer array
    Index ns2;
    T *const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_s1; //!< elements in a full src1 chunk
    Index elems_s2; //!< elements in a full src2 chunk
    Index elems_dst; //!< elements in a full dst chunk
    Index m;
    Index n;
    Index k;
    Index mk; //!< m * k
    Scalar alpha;
    int shift_s1; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_s2; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Resolve a flat offset to its home pointer through the given chunk table.
template<typename T>
static __device__ __forceinline__ T const *chunked_s1_ptr(
    const T *const *sh, Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return sh[c] + (f - c * long(elems));
}

template<typename T>
static __device__ __forceinline__ T const *chunked_s2_ptr(
    const T *const *sh, Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return sh[c] + (f - c * long(elems));
}

template<typename T>
static __device__ __forceinline__ T *chunked_dst_ptr(
    T *const *sh, Index elems, int shift, long f) noexcept
{
    long const c = chunk_index(f, elems, shift);
    return sh[c] + (f - c * long(elems));
}

//! Chunk-aware generic multiply_fiber kernel: exactly the classic
//! cuda_kernel geometry ((i0, i1, i2) thread assignment), but src1 is
//! resolved at its fiber index i2 and src2/dst at their flat offset
//! (i1*k + i2)*m + i0 = i0 + m*i2 + m*k*i1 through the chunk tables. The
//! per-element arithmetic is exactly the classic kernel's, so the result is
//! bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(ChunkTables<T> t)
{
    Index i0 = threadIdx.x + blockIdx.x*blockDim.x,
          i1 = threadIdx.y + blockIdx.y*blockDim.y,
          i2 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    if(i0 < t.m and i1 < t.n and i2 < t.k)
    {
        const Y src1_val =
            alpha * Y{*chunked_s1_ptr(t.src1, t.elems_s1, t.shift_s1, i2)};
        long const f = (long(i1)*t.k + i2)*t.m + i0;
        T *dp = chunked_dst_ptr(t.dst, t.elems_dst, t.shift_dst, f);
        *dp = T{src1_val
            * Y{*chunked_s2_ptr(t.src2, t.elems_s2, t.shift_s2, f)}};
    }
}

//! Chunk-aware m==1 multiply_fiber kernel: exactly the classic
//! cuda_kernel_m1 geometry, but each src1/src2/dst element is resolved
//! through its chunk table. The per-element arithmetic is exactly the
//! classic kernel's, so the result is bitwise cuda()'s m==1 branch.
template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void chunked_cuda_kernel_m1(ChunkTables<T> t)
{
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (t.k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    using Y = typename T::repr_t;
    const Y alpha{t.alpha};
    __shared__ Y src1_block[BLOCK_ROW];
    __shared__ T src2_block[BLOCK_ROW][BLOCK_COL+1];
    Index src1_l = dst_block_l*BLOCK_ROW + threadIdx.x;
    if(src1_l < t.k and threadIdx.x < BLOCK_ROW)
    {
        src1_block[threadIdx.x] = alpha *
            static_cast<Y>(*chunked_s1_ptr(t.src1, t.elems_s1, t.shift_s1, src1_l));
    }
    __syncthreads();
    if(global_dst_l < t.k)
    {
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        // Classic two-phase staging (load src2 into shared, then write
        // dst); src2_block[dst_l] is thread-private for this launch
        // geometry, so no barrier sits between the phases, exactly like
        // the classic kernel. Piece 4: this thread's columns live at the
        // flat offsets fe0 + c*k, a per-thread walk; resolve each
        // operand's home pointer ONCE through its virtual contiguous
        // base q = chunk_base - chunk*elems (the element at flat f is
        // q[f]) and rebase only when the walk crosses a chunk boundary
        // -- the per-element table resolution (a 64-bit divmod whenever
        // the chunk size is not a power of two, which is every 1-chunk
        // operand) sat in the middle of both column loops. Same elements
        // in the same order, so the result is unchanged bitwise.
        Index const ncols = (dst_block_j+1)*BLOCK_COL <= t.n
            ? Index(BLOCK_COL) : t.n - global_dst_j;
        long const fe0 = long(global_dst_l) + long(global_dst_j)*long(t.k);
        long const step_col = long(BLOCK_COL_STEP) * long(t.k);
        long e2 = t.elems_s2, ed = t.elems_dst;
        // (rule 7) span-start chunk residency by shift/mask on the
        // per-operand shifts (class shift or the 62 sentinel), never
        // a runtime divisor.
        int const sh2s = t.shift_s2, shds = t.shift_dst;
        // Single-run fast path: the whole column range inside one chunk
        // of BOTH src2 and dst (every 1-chunk operand, and interior
        // spans) -> the classic two-phase loops verbatim through the
        // virtual contiguous bases (the element at flat f is q[f]).
        long const fe_hi = fe0 + long(ncols - 1) * long(t.k);
        long const c20 = fe0 >> sh2s, cd0 = fe0 >> shds;
        if(fe_hi >> sh2s == c20 && fe_hi >> shds == cd0)
        {
            T const *q2 = t.src2[c20] - c20 * e2;
            T *qdd = t.dst[cd0] - cd0 * ed;
            for(int c = 0; c < ncols; c += BLOCK_COL_STEP)
            {
                src2_block[dst_l][c] = q2[fe0 + long(c) * long(t.k)];
            }
            for(int c = 0; c < ncols; c += BLOCK_COL_STEP)
            {
                qdd[fe0 + long(c) * long(t.k)] =
                    static_cast<T>(src1_block[dst_l] *
                        static_cast<Y>(src2_block[dst_l][c]));
            }
        }
        else
        for(int phase = 0; phase < 2; ++phase)
        {
            long c2 = fe0 >> sh2s;
            long off2 = fe0 - c2 * e2;
            T const *q2 = t.src2[c2] - c2 * e2;
            long cd = fe0 >> shds;
            long offd = fe0 - cd * ed;
            T *qdd = t.dst[cd] - cd * ed;
            Index c = 0;
            long fe = fe0;
            while(c < ncols)
            {
                // Columns before the next chunk boundary of either operand.
                long const room2 = (e2 - off2 + step_col - 1) / step_col;
                long const roomd = (ed - offd + step_col - 1) / step_col;
                long const run = (room2 < roomd ? room2 : roomd)
                    * BLOCK_COL_STEP;
                Index const run_cols = run < long(ncols - c)
                    ? Index(run) : ncols - c;
                for(Index i = 0; i < run_cols; i += BLOCK_COL_STEP)
                {
                    if(phase == 0)
                    {
                        src2_block[dst_l][c] = q2[fe];
                    }
                    else
                    {
                        qdd[fe] = static_cast<T>(src1_block[dst_l] *
                            static_cast<Y>(src2_block[dst_l][c]));
                    }
                    fe += step_col;
                    off2 += step_col;
                    offd += step_col;
                    c += BLOCK_COL_STEP;
                }
                if(off2 >= e2)
                {
                    c2 = fe >> sh2s;
                    off2 = fe - c2 * e2;
                    q2 = t.src2[c2] - c2 * e2;
                }
                if(offd >= ed)
                {
                    cd = fe >> shds;
                    offd = fe - cd * ed;
                    qdd = t.dst[cd] - cd * ed;
                }
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
    const T *const *src1_chunks, Index ns1, Index elems_s1,
    const T *const *src2_chunks, Index ns2, Index elems_s2,
    T *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || ns1 <= 0 || ns2 <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.src1 = src1_chunks;
    t.ns1 = ns1;
    t.src2 = src2_chunks;
    t.ns2 = ns2;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_s1 = elems_s1;
    t.elems_s2 = elems_s2;
    t.elems_dst = elems_dst;
    t.m = m;
    t.n = n;
    t.k = k;
    t.mk = m * k;
    t.alpha = alpha;
    t.shift_s1 = chunk_shift_of<T>(t.ns1, elems_s1, "nntile_multiply_fiber");
    t.shift_s2 = chunk_shift_of<T>(t.ns2, elems_s2, "nntile_multiply_fiber");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_multiply_fiber");
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (chunked_cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0,
            stream>>>(t);
    }
    else
    {
        dim3 threads(std::min(int(m), 8), std::min(int(n), 8),
                std::min(int(k), 16));
        dim3 blocks((m+threads.x-1)/threads.x, (n+threads.y-1)/threads.y,
                (k+threads.z-1)/threads.z);
        (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
    }
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *src1, const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *src1, const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *src1, const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *src1, const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp32_t *const *src1_chunks, Index ns1, Index elems_s1,
        const fp32_t *const *src2_chunks, Index ns2, Index elems_s2,
        fp32_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp64_t *const *src1_chunks, Index ns1, Index elems_s1,
        const fp64_t *const *src2_chunks, Index ns2, Index elems_s2,
        fp64_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const bf16_t *const *src1_chunks, Index ns1, Index elems_s1,
        const bf16_t *const *src2_chunks, Index ns2, Index elems_s2,
        bf16_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Scalar alpha,
        const fp16_t *const *src1_chunks, Index ns1, Index elems_s1,
        const fp16_t *const *src2_chunks, Index ns2, Index elems_s2,
        fp16_t *const *dst_chunks, Index ndst, Index elems_dst)
    noexcept;






} // namespace nntile::kernel::multiply_fiber
