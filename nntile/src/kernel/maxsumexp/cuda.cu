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
        Index k, Index mk, const T * __restrict__ src, Scalar beta,
        T * __restrict__ maxsumexp)
{
    Index i1_block = blockIdx.y, i2_block = blockIdx.z,
          i0_start = threadIdx.x, i0_step = blockDim.x;

    using Y = typename T::repr_t;
    constexpr Y zero = 0.0, one = 1.0;
    const bool overwrite = (beta == 0.0);
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
                // Update output
                if(i0_start == 0)
                {
                    Index dst_offset = i1 + i2*m;
                    if(block_sum_val > 0)
                    {
                        max_val = block_max_val;
                        sum_val = block_sum_val;
                        if(overwrite)
                        {
                            maxsumexp[2*dst_offset] = T{max_val};
                            maxsumexp[2*dst_offset+1] = T{sum_val};
                        }
                        else
                        {
                            Y max_output;
                            Y sum_output = Y{maxsumexp[2*dst_offset+1]};
                            if(sum_output == zero)
                            {
                                max_output = max_val;
                                sum_output = sum_val;
                            }
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
                    else if(overwrite)
                    {
                        maxsumexp[2*dst_offset] = T{zero};
                        maxsumexp[2*dst_offset+1] = T{zero};
                    }
                }
            }
        }
    }
}

template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void cuda_kernel_m1(Index n, Index k, const T *src, Scalar beta, T *dst)
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    constexpr Y one = 1.0;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    const bool overwrite = (beta == 0.0);
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
            if(overwrite)
            {
                dst[2*blockIdx.x] = static_cast<T>(max2);
                dst[2*blockIdx.x+1] = static_cast<T>(sumexp2);
            }
            else
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
        else if(overwrite)
        {
            dst[2*blockIdx.x] = static_cast<T>(0.0);
            dst[2*blockIdx.x+1] = static_cast<T>(0.0);
        }
    }
}

template <typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, const T *src,
          Scalar beta, T *maxsumexp)
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
                    k, src, beta, maxsumexp);
        }
        else if(k <= 2048)
        {
            dim3 threads(128);
            dim3 blocks(n);
            (cuda_kernel_m1<T, 2048, 16>)<<<blocks, threads, 0, stream>>>(n,
                    k, src, beta, maxsumexp);
        }
        else
        {
            dim3 threads(256);
            dim3 blocks(n);
            (cuda_kernel_m1<T, 4096, 16>)<<<blocks, threads, 0, stream>>>(n,
                    k, src, beta, maxsumexp);
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
                n_per_block, k, m*k, src, beta, maxsumexp);
    }
}

// ---------------------------------------------------------------------------
// One-launch chunk-aware variant (see cuda.hh).  The chunk base-pointer
// tables live in device memory (the driver uploads them through the shared
// ring).  The arithmetic is copied verbatim from cuda_kernel/cuda_kernel_m1:
// same thread partition over the reduction axis, same max/atomicAdd combine,
// same m==1 tree.  Only the element address changes (flat offset -> chunk
// index + in-chunk offset), so the result is the contiguous kernel's up to
// its own shared-atomic ordering floor.
// ---------------------------------------------------------------------------

template<typename T>
struct MaxSumExpChunkTables
{
    const T* const* src;
    Index nsrc;
    Index elems_src;
    int shift_src;
    long total_src;
    T* const* dst;
    Index ndst;
    Index elems_dst;
    int shift_dst;
    long total_dst;
};

static __device__ __forceinline__ Index mse_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

static __device__ __forceinline__ long mse_off(long f, Index elems, int shift,
    Index c)
{
    return f & ((1LL << shift) - 1);
}

template<typename T>
static __device__ __forceinline__ T mse_src_at(
    MaxSumExpChunkTables<T> const &t, long f)
{
    Index const c = mse_chunk_of(f, t.elems_src, t.shift_src);
    return t.src[c][mse_off(f, t.elems_src, t.shift_src, c)];
}

template<typename T>
static __device__ __forceinline__ T mse_dst_get(
    MaxSumExpChunkTables<T> const &t, long f)
{
    Index const c = mse_chunk_of(f, t.elems_dst, t.shift_dst);
    return t.dst[c][mse_off(f, t.elems_dst, t.shift_dst, c)];
}

template<typename T>
static __device__ __forceinline__ void mse_dst_set(
    MaxSumExpChunkTables<T> const &t, long f, T v)
{
    Index const c = mse_chunk_of(f, t.elems_dst, t.shift_dst);
    t.dst[c][mse_off(f, t.elems_dst, t.shift_dst, c)] = v;
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index m, Index m_per_block, Index n,
        Index n_per_block, Index k, Scalar beta, MaxSumExpChunkTables<T> t)
{
    Index i1_block = blockIdx.y, i2_block = blockIdx.z,
          i0_start = threadIdx.x, i0_step = blockDim.x;

    using Y = typename T::repr_t;
    constexpr Y zero = 0.0, one = 1.0;
    const bool overwrite = (beta == 0.0);
    if(i0_start < k)
    {
        for(Index i1 = i1_block*m_per_block;
                i1 < (i1_block+1)*m_per_block and i1 < m; ++i1)
        {
            for(Index i2 = i2_block*n_per_block;
                    i2 < (i2_block+1)*n_per_block and i2 < n; ++i2)
            {
                long const base = long(i1) + long(m)*long(k)*long(i2);
                // Whole-fiber fast path: every real GPT-2 fiber lies inside
                // one 4 MiB chunk, so hoist the chunk lookup out of the
                // reduction loop (the per-element lookup was the residual).
                long const fib_end = base + long(m)*(long(k)-1) + 1;
                Index const cf = mse_chunk_of(base, t.elems_src,
                    t.shift_src);
                long const cf_end = (long(cf)+1)*long(t.elems_src)
                    < t.total_src
                    ? (long(cf)+1)*long(t.elems_src) : t.total_src;
                bool const fib_one = fib_end <= cf_end;
                T const *fib_ptr = t.src[cf];
                long const fib_off = base - long(cf)*long(t.elems_src);
                Y max_val = Y{fib_one
                    ? fib_ptr[fib_off + long(i0_start)*m]
                    : mse_src_at(t, base + long(i0_start)*m)};
                Y sum_val = one;
                for(Index i0 = i0_start+i0_step; i0 < k; i0 += i0_step)
                {
                    Y val = Y{fib_one
                        ? fib_ptr[fib_off + long(i0)*m]
                        : mse_src_at(t, base + long(i0)*m)};
                    if(::isinf(val))
                    {
                        continue;
                    }
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
                volatile __shared__ Y block_max_val;
                __shared__ Y block_sum_val;
                if(i0_start == 0)
                {
                    block_max_val = max_val;
                    block_sum_val = zero;
                }
                __syncthreads();
                while(block_max_val < max_val)
                {
                    block_max_val = max_val;
                }
                __syncthreads();
                if(not ::isinf(max_val))
                {
                    sum_val *= ::exp(max_val - block_max_val);
                    atomicAdd(&block_sum_val, sum_val);
                }
                __syncthreads();
                if(i0_start == 0)
                {
                    long const dst_f = 2*(long(i1) + long(m)*long(i2));
                    if(block_sum_val > 0)
                    {
                        max_val = block_max_val;
                        sum_val = block_sum_val;
                        if(overwrite)
                        {
                            mse_dst_set(t, dst_f, T{max_val});
                            mse_dst_set(t, dst_f+1, T{sum_val});
                        }
                        else
                        {
                            Y max_output;
                            Y sum_output = Y{mse_dst_get(t, dst_f+1)};
                            if(sum_output == zero)
                            {
                                max_output = max_val;
                                sum_output = sum_val;
                            }
                            else
                            {
                                max_output = Y{mse_dst_get(t, dst_f)};
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
                            mse_dst_set(t, dst_f, T{max_output});
                            mse_dst_set(t, dst_f+1, T{sum_output});
                        }
                    }
                    else if(overwrite)
                    {
                        mse_dst_set(t, dst_f, T{zero});
                        mse_dst_set(t, dst_f+1, T{zero});
                    }
                }
            }
        }
    }
}

//! Whole-buffer m==1 kernel: one block per panel and the contiguous
//! kernel's reduction structure verbatim (so src_val stays in registers).
//! Piece 7: panels that straddle a chunk boundary no longer return early
//! for a separate straddle kernel (a second launch whose 12-block grid
//! serialized AFTER the main wave cost 30 us on the 196.32 MiB GPT-2
//! logits shape = the pair-4 5% residual); the SAME block now takes the
//! per-element chunk-resolved read path in place, overlapping with the
//! whole-panel blocks inside the ONE launch (rule 4). The per-thread read
//! sequence of both paths is the contiguous kernel's own, so the result
//! stays bitwise.
template<typename T, int BLOCK_ROW, int LOOP>
static __global__
void chunked_cuda_kernel_m1_whole(Index n, Index k, Scalar beta,
        MaxSumExpChunkTables<T> t)
{
    Index src_l_block_end = (k/BLOCK_ROW) * BLOCK_ROW;
    using Y = typename T::repr_t;
    constexpr Y one = 1.0;
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    const bool overwrite = (beta == 0.0);
    volatile __shared__ Y dst_block_max[BLOCK_ROW_STEP];
    volatile __shared__ Y dst_block_sumexp[BLOCK_ROW_STEP];
    (void)n;
    long const panel_base = long(blockIdx.x)*long(k);
    Index const cp = mse_chunk_of(panel_base, t.elems_src, t.shift_src);
    long const cp_end = (long(cp)+1)*long(t.elems_src) < t.total_src
        ? (long(cp)+1)*long(t.elems_src) : t.total_src;
    bool const whole = panel_base + long(k) <= cp_end;
    const T *src = t.src[cp] + (panel_base - long(cp)*long(t.elems_src));
    Y dst_max=-INFINITY, dst_sumexp=0.0;
    T src_val[LOOP];
    if(whole)
    {
        // The whole-panel path is the contiguous kernel's load stream
        // verbatim, under a block-uniform branch (NOT a per-load select:
        // the select form cost the 48 MiB 3-chunk control ~1.5%).
        for(Index src_l = threadIdx.x; src_l < src_l_block_end;
                src_l += BLOCK_ROW)
        {
            const T *src_fiber = src + src_l;
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
        Index src_l = threadIdx.x + src_l_block_end;
        int c_max = (k-src_l+BLOCK_ROW_STEP-1) / BLOCK_ROW_STEP;
        const T *src_fiber = src + src_l;
        // The tail is read straight into the update: it never indexes the
        // batched array with a runtime bound (that spilled the array to
        // local memory and doubled the kernel time), and when
        // k == BLOCK_ROW it is empty.
        for(int c = 0; c < c_max; ++c)
        {
            Y val = static_cast<Y>(src_fiber[c*BLOCK_ROW_STEP]);
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
    else
    {
        // The straddling panel: per-element chunk-resolved reads (the
        // deleted straddle kernel's body, verbatim -- same per-thread
        // read sequence, so the reduction stays bitwise).
        for(Index src_l = threadIdx.x; src_l < src_l_block_end;
                src_l += BLOCK_ROW)
        {
            for(int c = 0; c < LOOP; ++c)
            {
                src_val[c] = mse_src_at(t, panel_base + long(src_l)
                    + long(c)*BLOCK_ROW_STEP);
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
        Index src_l = threadIdx.x + src_l_block_end;
        int c_max = (k-src_l+BLOCK_ROW_STEP-1) / BLOCK_ROW_STEP;
        for(int c = 0; c < c_max; ++c)
        {
            src_val[c] = mse_src_at(t, panel_base + long(src_l)
                + long(c)*BLOCK_ROW_STEP);
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
    }
    dst_block_max[threadIdx.x] = dst_max;
    dst_block_sumexp[threadIdx.x] = dst_sumexp;
    __syncthreads();
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
    if(threadIdx.x == 0)
    {
        Y max2 = dst_block_max[0];
        Y sumexp2 = dst_block_sumexp[0];
        long const dst_f = 2*long(blockIdx.x);
        if(not ::isinf(max2))
        {
            if(overwrite)
            {
                mse_dst_set(t, dst_f, static_cast<T>(max2));
                mse_dst_set(t, dst_f+1, static_cast<T>(sumexp2));
            }
            else
            {
                Y max = static_cast<Y>(mse_dst_get(t, dst_f));
                Y sumexp = static_cast<Y>(mse_dst_get(t, dst_f+1));
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
                mse_dst_set(t, dst_f, static_cast<T>(max));
                mse_dst_set(t, dst_f+1, static_cast<T>(sumexp));
            }
        }
        else if(overwrite)
        {
            mse_dst_set(t, dst_f, static_cast<T>(0.0));
            mse_dst_set(t, dst_f+1, static_cast<T>(0.0));
        }
    }
}

template <typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst,
        long total_src, long total_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || nsrc <= 0 || ndst <= 0
        || elems_src <= 0 || elems_dst <= 0)
    {
        return;
    }
    MaxSumExpChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.elems_src = elems_src;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_maxsumexp");
    t.total_src = total_src > 0 ? total_src : long(elems_src)*long(nsrc);
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_dst = elems_dst;
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_maxsumexp");
    t.total_dst = total_dst > 0 ? total_dst
        : long(elems_dst)*long(ndst);
    if(m == 1)
    {
        // ONE launch over all n panels (piece 7): a straddling panel's
        // block takes the per-element chunk-resolved path inside this
        // same kernel, so no host-side straddle list, no second launch.
        dim3 threads(64);
        dim3 blocks(n);
        if(k > 1024)
        {
            threads = dim3(128);
        }
        if(k > 2048)
        {
            threads = dim3(256);
        }
        if(k <= 1024)
        {
            (chunked_cuda_kernel_m1_whole<T, 1024, 16>)<<<blocks, threads,
                0, stream>>>(n, k, beta, t);
        }
        else if(k <= 2048)
        {
            (chunked_cuda_kernel_m1_whole<T, 2048, 16>)<<<blocks, threads,
                0, stream>>>(n, k, beta, t);
        }
        else
        {
            (chunked_cuda_kernel_m1_whole<T, 4096, 16>)<<<blocks, threads,
                0, stream>>>(n, k, beta, t);
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
        (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m,
                m_per_block, n, n_per_block, k, beta, t);
    }
}

template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t *src, Scalar beta, fp32_t *maxsumexp)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t *src, Scalar beta, fp64_t *maxsumexp)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t *src, Scalar beta, bf16_t *maxsumexp)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp16_t *src, Scalar beta, fp16_t *maxsumexp)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp32_t* const* src_chunks, Index nsrc, Scalar beta,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, long total_src, long total_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp64_t* const* src_chunks, Index nsrc, Scalar beta,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, long total_src, long total_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const bf16_t* const* src_chunks, Index nsrc, Scalar beta,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, long total_src, long total_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        const fp16_t* const* src_chunks, Index nsrc, Scalar beta,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst, long total_src, long total_dst) noexcept;

} // namespace nntile::kernel::maxsumexp
