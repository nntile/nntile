/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/softmax_inplace/cuda.cu
 * Inplace softmax operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/softmax_inplace/cuda.hh"
#include <stdio.h>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::softmax_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index m_per_block, Index n, Index n_per_block,
        Index k, const T *maxsumexp, Scalar alpha, T *dst)
{
    Index i0_block = blockIdx.y, i1_block = blockIdx.z,
          i2_start = threadIdx.x, i2_step = blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};

    for(Index i0 = i0_block*m_per_block;
            i0 < (i0_block+1)*m_per_block and i0 < m; ++i0)
    {
        for(Index i1 = i1_block*n_per_block;
                i1 < (i1_block+1)*n_per_block and i1 < n; ++i1)
        {
            Index dst_offset = i1*k*m + i0;
            T *dst_slice = dst + dst_offset;
            // Max and sum of exponents
            __shared__ Y max, sum;
            if(i2_start == 0)
            {
                Index src_offset = m*i1 + i0;
                max = Y{maxsumexp[2*src_offset]};
                sum = Y{maxsumexp[2*src_offset+1]};
            }
            __syncthreads();
            for(Index i2 = i2_start; i2 < k; i2 += i2_step)
            {
                // Value-to-update
                T &val = dst_slice[i2*m];
                // Update value
                if(not ::isinf(Y{val}))
                {
                    val = T{alpha * ::exp(Y{val}-max) / sum};
                }
                else
                {
                    val = T{zero};
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, const T *maxsumexp, Scalar alpha_,
        T *dst)
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
        T *dst_fiber = dst + global_dst_l + global_dst_j*k;
        constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
            {
                dst_block[dst_l][c] = dst_fiber[c*k];
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
                dst_block[dst_l][c] = dst_fiber[c*k];
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
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *maxsumexp,
        Scalar alpha, T *dst)
    noexcept
//! softmax of a buffer along middle axis
/*!
 *
 * @param[in] m: Size of the first mode of dst and sumexp arrays
 * @param[in] n: Size of the last mode of dst and sumexp arrays
 * @param[in] k: Size of the middle mode of dst array
 * @param[in] maxsumexp: Maximums and sums of exponents of slices
 * @param[in] alpha: Scalar multiplier for the output
 * @param[in] dst: Contiguous output array
 * */
{
    // Source is an m-by-n matrix and destination is an m-by-k-by-n tensor
    // Both source and destination are Fortran-contiguous
    // Custom case m==1
    if(m == 1)
    {
        dim3 threads(256);
        dim3 blocks(((k+255)/256) * ((n+7)/8));
        (cuda_kernel_m1<T, 256, 8, 8>)<<<blocks, threads, 0, stream>>>(n, k,
                maxsumexp, alpha, dst);
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
                n_per_block, k, maxsumexp, alpha, dst);
    }
}

//! Chunk descriptors for the one-launch chunked softmax_inplace (cuda.hh).
template<typename T>
struct SoftmaxInplaceChunkTables
{
    const T *const *mse; //!< device view of the host maxsumexp pointer array
    Index nmse;
    T *const *dst; //!< device view of the host dst pointer array
    Index ndst;
    Index elems_mse;
    Index elems_dst;
    int shift_mse;
    int shift_dst;
    Index m;
    Index n;
    Index k;
    Scalar alpha;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ Index smi_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! In-chunk offset of a flat element offset (mask for a power of two).
static __device__ __forceinline__ long smi_off_of(long f, Index elems,
    int shift)
{
    return f & ((1LL << shift) - 1);
}

//! Read element f of an operand through its shared chunk table.
template<typename T>
static __device__ __forceinline__ T smi_get(const T *const *sh, Index elems,
    int shift, long f)
{
    Index const c = smi_chunk_of(f, elems, shift);
    return sh[c][smi_off_of(f, elems, shift)];
}

//! Write element f of an operand through its shared chunk table.
template<typename T>
static __device__ __forceinline__ void smi_set(T *const *sh, Index elems,
    int shift, long f, T v)
{
    Index const c = smi_chunk_of(f, elems, shift);
    sh[c][smi_off_of(f, elems, shift)] = v;
}

//! Chunk-aware elementwise softmax_inplace, one contiguous span of `span`
//! elements per block, with the block's maxsumexp pairs read once into shared
//! memory (see the softmax kernel). Piece 9 (rule 10): the chunk-pointer
//! tables are read straight from the device ring (no staging, no table
//! __syncthreads); the dynamic shared memory carries ONLY the per-row
//! (max, sum) pairs, whose barrier is algorithmic. Bitwise cuda()'s result.
template<typename T>
static __global__
void chunked_cuda_kernel_staged(SoftmaxInplaceChunkTables<T> t, long total,
    long span)
{
    extern __shared__ char smi_smem[];
    using Y = typename T::repr_t;
    Y *sh_max = reinterpret_cast<Y*>(smi_smem);
    long const mk = long(t.m) * long(t.k);
    long const nent = long(t.m) * (span/mk + 2);
    Y *sh_sum = sh_max + nent;
    long const f0 = long(blockIdx.x)*span;
    long const f1 = f0+span < total ? f0+span : total;
    long const z0 = f0/mk;
    long const z1 = (f1-1)/mk;
    long const msz = long(t.m)*(z1-z0+1);
    for(long j = threadIdx.x; j < msz; j += blockDim.x)
    {
        long const fm = 2*(long(t.m)*z0 + j);
        sh_max[j] = Y{smi_get(t.mse, t.elems_mse, t.shift_mse, fm)};
        sh_sum[j] = Y{smi_get(t.mse, t.elems_mse, t.shift_mse, fm+1)};
    }
    __syncthreads();
    long const span_b = f1 - f0;
    Index const cd = smi_chunk_of(f0, t.elems_dst, t.shift_dst);
    long const od = smi_off_of(f0, t.elems_dst, t.shift_dst);
    bool const one_dst = (od + span_b) <= long(t.elems_dst);
    T *dp = one_dst ? t.dst[cd] + od : nullptr;
    for(long f = f0 + threadIdx.x; f < f1; f += blockDim.x)
    {
        long const rel = f - f0;
        long const z = f/mk;
        long const r = f % long(t.m);
        long const idx = long(t.m)*(z-z0) + r;
        Y const max = sh_max[idx];
        Y const sum = sh_sum[idx];
        Y val = Y{one_dst ? dp[rel]
            : smi_get(t.dst, t.elems_dst, t.shift_dst, f)};
        T out;
        if(not ::isinf(val))
        {
            out = static_cast<T>(Y{t.alpha} * ::exp(val-max) / sum);
        }
        else
        {
            out = static_cast<T>(0.0);
        }
        if(one_dst)
        {
            dp[rel] = out;
        }
        else
        {
            smi_set(t.dst, t.elems_dst, t.shift_dst, f, out);
        }
    }
}

//! Chunk-aware m==1 softmax_inplace: cuda_kernel_m1's geometry and
//! instruction stream (BLOCK_ROW threads along the middle axis, BLOCK_COL
//! rows per block, the two-phase value staging through the shared
//! dst_block), with the chunk pointers resolved by a per-thread SEGMENT
//! WALK in registers: each (block, row) 256-element run of the middle
//! axis either lies inside one dst chunk (the common case -- its base
//! pointer is resolved by ONE ring-table read per (thread, row) and the
//! hot loops then address it exactly like the classic kernel's dst_fiber)
//! or the block falls to the per-element chunk-resolved path (only the
//! one block per row whose run actually contains a chunk boundary).
//! Rule 10: no chunk-TABLE staging and no pointer ever transits shared
//! memory -- the shared arrays are the classic kernel's own algorithmic
//! (max, sum) row staging and dst_block value staging. Every element
//! performs exactly the operations cuda_kernel_m1 performs, so the result
//! is bitwise cuda()'s.
template<typename T, int BLOCK_ROW, int BLOCK_COL, int BLOCK_LOOP>
static __global__
void chunked_cuda_kernel_m1(SoftmaxInplaceChunkTables<T> t)
{
    using Y = typename T::repr_t;
    __shared__ Y max_block[BLOCK_COL];
    __shared__ Y sumexp_block[BLOCK_COL];
    __shared__ T dst_block[BLOCK_ROW][BLOCK_COL];
    Index const n = t.n, k = t.k;
    Index dst_l = threadIdx.x % BLOCK_ROW;
    Index dst_griddim_row = (k+BLOCK_ROW-1) / BLOCK_ROW;
    Index dst_block_l = blockIdx.x % dst_griddim_row;
    Index dst_block_j = blockIdx.x / dst_griddim_row;
    Index global_dst_l = dst_l + dst_block_l*BLOCK_ROW;
    Index global_dst_j = dst_block_j*BLOCK_COL;
    Index ncols = n - global_dst_j;
    ncols = ncols < BLOCK_COL ? ncols : BLOCK_COL;
    Index const l0 = dst_block_l*BLOCK_ROW;
    Index const seg = l0+BLOCK_ROW <= k ? Index(BLOCK_ROW) : k-l0;
    // Stage the per-row (max,sum) pairs, exactly the classic kernel's own
    // algorithmic row staging (one barrier, no address computation).
    if(threadIdx.x < BLOCK_COL and threadIdx.x < ncols)
    {
        Index const j = global_dst_j + threadIdx.x;
        long const mi = 2*long(j);
        max_block[threadIdx.x] =
            Y{smi_get(t.mse, t.elems_mse, t.shift_mse, mi)};
        sumexp_block[threadIdx.x] =
            Y{smi_get(t.mse, t.elems_mse, t.shift_mse, mi+1)};
    }
    __syncthreads();
    if(global_dst_l < k)
    {
        if((dst_block_j+1)*BLOCK_COL <= n)
        {
            // dst segment walk: run base per (row) in registers, one
            // ring-table read per row; strad flags the (rare) block whose
            // run crosses a chunk boundary.
            bool strad = false;
            long const mmask = (1LL << t.shift_dst) - 1;
            long sc = long(global_dst_j)*long(k) + l0;
            T *dp[BLOCK_COL];
            #pragma unroll
            for(int c = 0; c < BLOCK_COL; ++c)
            {
                Index const ci = Index(sc >> t.shift_dst);
                strad = strad or ((long(ci)+1) << t.shift_dst)
                    < sc+long(seg);
                dp[c] = t.dst[ci] + (sc & mmask);
                sc += long(k);
            }
            if(not strad)
            {
                Index const rel = dst_l;
                const Y alpha{t.alpha};
                constexpr Y zero{0.0};
                constexpr int BLOCK_COL_STEP = BLOCK_COL / BLOCK_LOOP;
                #pragma unroll
                for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
                {
                    dst_block[dst_l][c] = dp[c][rel];
                }
                #pragma unroll
                for(Index c = 0; c < BLOCK_COL; c += BLOCK_COL_STEP)
                {
                    Y val = static_cast<Y>(dst_block[dst_l][c]);
                    if(not ::isinf(val))
                    {
                        val = alpha * ::exp(val-max_block[c])
                            / sumexp_block[c];
                        dp[c][rel] = static_cast<T>(val);
                    }
                    else
                    {
                        dp[c][rel] = static_cast<T>(zero);
                    }
                }
                return;
            }
        }
        // Tail block-col or straddling segment: per-element chunk-resolved
        // addressing (same read/write sequence, so still bitwise).
        long const f0e = long(global_dst_l)
            + long(global_dst_j)*long(k);
        for(Index c = 0; c < ncols; ++c)
        {
            long const f = f0e + long(c)*long(k);
            Y val = Y{smi_get(t.dst, t.elems_dst, t.shift_dst, f)};
            T out;
            if(not ::isinf(val))
            {
                out = static_cast<T>(Y{t.alpha} * ::exp(
                    val-max_block[c]) / sumexp_block[c]);
            }
            else
            {
                out = static_cast<T>(0.0);
            }
            smi_set(t.dst, t.elems_dst, t.shift_dst, f, out);
        }
    }
}

//! Fallback chunk-aware elementwise softmax_inplace (grid-stride): reads the
//! mse pair per element; used only for shapes whose per-block unit span would
//! not fit shared memory. Piece 9 (rule 10): direct ring reads.
template<typename T>
static __global__
void chunked_cuda_kernel_naive(SoftmaxInplaceChunkTables<T> t, long total)
{
    using Y = typename T::repr_t;
    long const mk = long(t.m) * long(t.k);
    long const stride = long(gridDim.x) * long(blockDim.x);
    for(long f = long(blockIdx.x)*blockDim.x + threadIdx.x; f < total;
            f += stride)
    {
        long const z = f / mk;
        long const r = f % long(t.m);
        long const mi = 2*(long(t.m)*z + r);
        Y max = Y{smi_get(t.mse, t.elems_mse, t.shift_mse, mi)};
        Y sum = Y{smi_get(t.mse, t.elems_mse, t.shift_mse, mi+1)};
        Y val = Y{smi_get(t.dst, t.elems_dst, t.shift_dst, f)};
        T out;
        if(not ::isinf(val))
        {
            out = static_cast<T>(Y{t.alpha} * ::exp(val-max) / sum);
        }
        else
        {
            out = static_cast<T>(0.0);
        }
        smi_set(t.dst, t.elems_dst, t.shift_dst, f, out);
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k,
        const T* const* mse_chunks, Index nmse, Scalar alpha,
        T* const* dst_chunks, Index ndst, Index elems_mse, Index elems_dst)
    noexcept
{
    if(m <= 0 or n <= 0 or k <= 0 or nmse <= 0 or ndst <= 0)
    {
        return;
    }
    SoftmaxInplaceChunkTables<T> t;
    t.mse = mse_chunks;
    t.nmse = nmse;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_mse = elems_mse;
    t.elems_dst = elems_dst;
    t.shift_mse = chunk_shift_of<T>(t.nmse, elems_mse, "nntile_softmax_inplace");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_softmax_inplace");
    t.m = m;
    t.n = n;
    t.k = k;
    t.alpha = alpha;
    long const total = long(m) * long(n) * long(k);
    if(total <= 0)
    {
        return;
    }
    if(m == 1)
    {
        // cuda()'s own m==1 geometry with the per-thread segment walk
        // (see the kernel comment): BLOCK_ROW threads along the middle
        // axis, BLOCK_COL rows per block, no per-element divmod. No
        // dynamic smem (rule 10: no table/pointer staging at all -- the
        // walk bases live in registers).
        constexpr int BLOCK_ROW = 256, BLOCK_COL = 8, BLOCK_LOOP = 8;
        dim3 threads(256);
        dim3 blocks(((k+BLOCK_ROW-1)/BLOCK_ROW)
                    * ((n+BLOCK_COL-1)/BLOCK_COL));
        (chunked_cuda_kernel_m1<T, BLOCK_ROW, BLOCK_COL, BLOCK_LOOP>)<<<
            blocks, threads, 0, stream>>>(t);
        return;
    }
    int const threads = 256;
    long const span = long(threads) * 8;
    long const mk = long(m) * long(k);
    long const mszmax = long(m) * (span/mk + 2);
    size_t const staged_bytes =
        size_t(mszmax) * sizeof(typename T::repr_t) * 2;
    if(staged_bytes <= 46000)
    {
        long const blocks = (total + span - 1) / span;
        (chunked_cuda_kernel_staged<T>)<<<
            dim3(static_cast<unsigned>(blocks)), threads,
            staged_bytes, stream>>>(t, total, span);
        return;
    }
    long const per = long(threads) * 8;
    long blocks = (total + per - 1) / per;
    if(blocks < 1)
    {
        blocks = 1;
    }
    (chunked_cuda_kernel_naive<T>)<<<dim3(static_cast<unsigned>(blocks)),
        threads, 0, stream>>>(t, total);
}

template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *maxsumexp, Scalar alpha, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *maxsumexp, Scalar alpha, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *maxsumexp, Scalar alpha, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp16_t *maxsumexp, Scalar alpha, fp16_t *dst)
    noexcept;

template
void cuda<fp32_fast_tf32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_fast_tf32_t *maxsumexp, Scalar alpha, fp32_fast_tf32_t *dst)
    noexcept;

template
void cuda<fp32_fast_fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_fast_fp16_t *maxsumexp, Scalar alpha, fp32_fast_fp16_t *dst)
    noexcept;

template
void cuda<fp32_fast_bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_fast_bf16_t *maxsumexp, Scalar alpha, fp32_fast_bf16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *const *mse_chunks, Index nmse, Scalar alpha,
        fp32_t *const *dst_chunks, Index ndst, Index elems_mse,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *const *mse_chunks, Index nmse, Scalar alpha,
        fp64_t *const *dst_chunks, Index ndst, Index elems_mse,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *const *mse_chunks, Index nmse, Scalar alpha,
        bf16_t *const *dst_chunks, Index ndst, Index elems_mse,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp16_t *const *mse_chunks, Index nmse, Scalar alpha,
        fp16_t *const *dst_chunks, Index ndst, Index elems_mse,
        Index elems_dst) noexcept;





} // namespace nntile::kernel::softmax_inplace
