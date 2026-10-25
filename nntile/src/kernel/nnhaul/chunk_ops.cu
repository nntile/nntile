/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file src/kernel/nnhaul/chunk_ops.cu
 * Chunk-addressed classic operations: the six NNHaul codelets whose
 * chunked CUDA drivers used to loop over chunks on the host now reach
 * the whole operand through a by-value chunk table in ONE launch.
 *
 * The chunk class is the compile-time kChunkClassBytes and a power of
 * two, so a flat element position p maps to chunk p >> shift and
 * offset p & mask - no runtime division on the address path. The
 * arithmetic of every kernel mirrors the stock nntile kernel it
 * replaces (same accumulation order, same conversions).
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/nnhaul/chunk_ops.hh"

#include <nnhaul/nnhaul.hh>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace nntile::kernel::nnhaul
{

using ::nnhaul::Buffer;
using ::nnhaul::kChunkClassBytes;
using ::nnhaul::ElemType;

namespace
{

// Same by-value ceiling as the other chunk kernels (a 512 MiB operand at
// the 4 MiB class); overflow is a caller defect, loud.
constexpr int kMaxTableChunks = 128;

// One operand's chunk table, by value in the kernel arguments.
//
// `base` is the global id of ptr[0]; a launch may cover a contiguous
// sub-range of a larger operand's chunks. `shift`/`mask` describe the
// power-of-two chunk geometry: c = pos >> shift, off = pos & mask
// (unchunked uses shift = 63, mask = ~0ull: c = 0, off = pos).
struct ChunkTable
{
    void *ptr[kMaxTableChunks];
    unsigned long long shift;
    unsigned long long mask;
    int base;
    int n;
};

constexpr int log2_u64(unsigned long long v)
{
    int s = 0;
    while ((1ull << s) < v)
    {
        ++s;
    }
    return s;
}

template <typename T>
__device__ __forceinline__ T *chunk_at(ChunkTable const &t,
    unsigned long long pos)
{
    unsigned long long const c = (pos >> t.shift)
        - static_cast<unsigned long long>(t.base);
    unsigned long long const off = pos & t.mask;
    return static_cast<T *>(t.ptr[c]) + off;
}

// Arithmetic representation of an element type (nntile's repr_t).
template <typename T> struct Repr
{
    using Y = T;
    static __device__ __forceinline__ Y to(T v) { return v; }
    static __device__ __forceinline__ T from(Y v) { return v; }
};

template <> struct Repr<__nv_bfloat16>
{
    using Y = float;
    static __device__ __forceinline__ Y to(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 from(Y v)
    {
        return __float2bfloat16(v);
    }
};

template <> struct Repr<__half>
{
    using Y = float;
    static __device__ __forceinline__ Y to(__half v)
    {
        return __half2float(v);
    }
    static __device__ __forceinline__ __half from(Y v)
    {
        return __float2half(v);
    }
};

// ---------------------------------------------------------------------------
// Elementwise: multiply_inplace
// ---------------------------------------------------------------------------

template <typename T>
__global__ void multiply_inplace_kernel(unsigned long long nelems,
    float alpha_f, ChunkTable src, ChunkTable dst)
{
    unsigned long long const p =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= nelems)
    {
        return;
    }
    using R = Repr<T>;
    using Y = typename R::Y;
    T *d = chunk_at<T>(dst, p);
    Y const a = static_cast<Y>(alpha_f);
    *d = R::from(a * R::to(*chunk_at<T>(src, p)) * R::to(*d));
}

// ---------------------------------------------------------------------------
// Elementwise: sgd_step
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sgd_step_kernel(long long num_iter,
    unsigned long long num_elems, float momentum_f, float lr_f,
    float weight_decay_f, float dampening_f, bool nesterov,
    ChunkTable grad, ChunkTable velocity, ChunkTable param)
{
    unsigned long long const p =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= num_elems)
    {
        return;
    }
    using R = Repr<T>;
    using Y = typename R::Y;
    Y const momentum = static_cast<Y>(momentum_f);
    Y const lr = static_cast<Y>(lr_f);
    Y const weight_decay = static_cast<Y>(weight_decay_f);
    Y const dampening = static_cast<Y>(dampening_f);
    T *p_ptr = chunk_at<T>(param, p);
    T *v_ptr = chunk_at<T>(velocity, p);
    Y const p_val = R::to(*p_ptr);
    Y grad_val = R::to(*chunk_at<T>(grad, p));
    if (weight_decay != Y{0})
    {
        grad_val += weight_decay * p_val;
    }
    if (momentum != Y{0})
    {
        Y velocity_val;
        if (num_iter == 1)
        {
            velocity_val = grad_val;
        }
        else
        {
            velocity_val = momentum * R::to(*v_ptr)
                + static_cast<Y>(1.0 - dampening) * grad_val;
        }
        *v_ptr = R::from(velocity_val);
        if (nesterov)
        {
            grad_val += momentum * velocity_val;
        }
        else
        {
            grad_val = velocity_val;
        }
    }
    *p_ptr = R::from(p_val - lr * grad_val);
}

// ---------------------------------------------------------------------------
// subtract_indexed_outputs: one block per output column
// ---------------------------------------------------------------------------

template <typename T>
__global__ void subtract_indexed_outputs_kernel(unsigned long long n_labels,
    unsigned long long n_outputs, long long ignore_index, float val_f,
    ChunkTable labels, ChunkTable dst)
{
    unsigned long long const i = blockIdx.x;
    if (i >= n_outputs)
    {
        return;
    }
    using R = Repr<T>;
    using Y = typename R::Y;
    long long const lab = *chunk_at<int64_t>(labels, i);
    if (lab == ignore_index)
    {
        unsigned long long const base = i * n_labels;
        for (unsigned long long l = threadIdx.x; l < n_labels;
            l += blockDim.x)
        {
            *chunk_at<T>(dst, base + l) = R::from(Y{0});
        }
        return;
    }
    if (threadIdx.x == 0)
    {
        T *d = chunk_at<T>(dst,
            static_cast<unsigned long long>(lab) + i * n_labels);
        Y const dst_val = R::to(*d);
        *d = R::from(dst_val - static_cast<Y>(val_f));
    }
}

// ---------------------------------------------------------------------------
// sum_fiber: same thread geometry and per-thread i1 order as the stock
// kernel, but each (i2, batch) output may read any chunk of src.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void sum_fiber_kernel(unsigned long long m,
    unsigned long long n, unsigned long long k, unsigned long long batch,
    float alpha_f, ChunkTable src, float beta_f, ChunkTable dst)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const i2_batched =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    unsigned long long const i2 = i2_batched % k;
    unsigned long long const b = i2_batched / k;
    unsigned long long const i0_start = threadIdx.y;
    unsigned long long const i0_step = blockDim.y;
    unsigned long long const i1_start = threadIdx.z;
    unsigned long long const i1_step = blockDim.z;
    Y const zero{0.0};
    Y sum = zero;
    if (b < batch)
    {
        for (unsigned long long i1 = i1_start; i1 < n; i1 += i1_step)
        {
            for (unsigned long long i0 = i0_start; i0 < m; i0 += i0_step)
            {
                unsigned long long const flat =
                    ((i1 + b * n) * k + i2) * m + i0;
                sum += R::to(*chunk_at<T>(src, flat));
            }
        }
        __shared__ Y block_sum[1];
        if (i1_start == 0 && i0_start == 0)
        {
            block_sum[threadIdx.x] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x], sum);
        __syncthreads();
        if (i1_start == 0 && i0_start == 0)
        {
            sum = block_sum[threadIdx.x];
            Y const alpha = static_cast<Y>(alpha_f);
            Y const beta = static_cast<Y>(beta_f);
            if (beta == zero)
            {
                sum *= alpha;
            }
            else
            {
                sum = beta * R::to(*chunk_at<T>(dst, i2_batched))
                    + alpha * sum;
            }
            *chunk_at<T>(dst, i2_batched) = R::from(sum);
        }
    }
}

// ---------------------------------------------------------------------------
// total_sum_accum: same block geometry (256 threads, one output per
// thread) and the kernel's own shared/global accumulation.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void total_sum_accum_kernel(float alpha_f,
    unsigned long long n_labels, unsigned long long n_outputs,
    long long ignore_index, ChunkTable logsumexp, ChunkTable src,
    ChunkTable labels, ChunkTable val)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const i =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    __shared__ Y block_val;
    if (threadIdx.x == 0)
    {
        block_val = Y{0};
    }
    __syncthreads();
    if (i < n_outputs)
    {
        Y const val1 = R::to(*chunk_at<T>(logsumexp, i));
        Y val2 = val1;
        long long const lab = *chunk_at<int64_t>(labels, i);
        if (lab != ignore_index)
        {
            val2 = R::to(*chunk_at<T>(src,
                static_cast<unsigned long long>(lab) + i * n_labels));
        }
        atomicAdd(&block_val, val1 - val2);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(chunk_at<float>(val, 0),
            static_cast<float>(static_cast<Y>(alpha_f) * block_val));
    }
}

// ---------------------------------------------------------------------------
// maxsumexp, m > 1 path: one block per (i0, i2) output, threads across
// the k reduction positions - identical to the stock kernel's mapping.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void maxsumexp_kernel(unsigned long long m,
    unsigned long long m_per_block, unsigned long long n,
    unsigned long long n_per_block, unsigned long long k, ChunkTable src,
    float beta_f, ChunkTable dst)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const i1_block = blockIdx.y;
    unsigned long long const i2_block = blockIdx.z;
    unsigned long long const i0_start = threadIdx.x;
    unsigned long long const i0_step = blockDim.x;
    Y const zero{0.0};
    Y const one{1.0};
    bool const overwrite = (static_cast<Y>(beta_f) == zero);
    if (i0_start >= k)
    {
        return;
    }
    for (unsigned long long i1 = i1_block * m_per_block;
        i1 < (i1_block + 1) * m_per_block && i1 < m; ++i1)
    {
        for (unsigned long long i2 = i2_block * n_per_block;
            i2 < (i2_block + 1) * n_per_block && i2 < n; ++i2)
        {
            unsigned long long const base = i2 * m * k + i1;
            Y max_val = R::to(*chunk_at<T>(src, base + i0_start * m));
            Y sum_val = one;
            for (unsigned long long i0 = i0_start + i0_step; i0 < k;
                i0 += i0_step)
            {
                Y const val = R::to(*chunk_at<T>(src, base + i0 * m));
                if (::isinf(val))
                {
                    continue;
                }
                if (max_val < val)
                {
                    sum_val = sum_val * (::exp(max_val - val)) + one;
                    max_val = val;
                }
                else
                {
                    sum_val += ::exp(val - max_val);
                }
            }
            volatile __shared__ Y block_max_val;
            __shared__ Y block_sum_val;
            if (i0_start == 0)
            {
                block_max_val = max_val;
                block_sum_val = zero;
            }
            __syncthreads();
            while (block_max_val < max_val)
            {
                block_max_val = max_val;
            }
            __syncthreads();
            if (!::isinf(max_val))
            {
                sum_val *= ::exp(max_val - block_max_val);
                atomicAdd(&block_sum_val, sum_val);
            }
            __syncthreads();
            if (i0_start == 0)
            {
                unsigned long long const dst_offset = i1 + i2 * m;
                if (block_sum_val > Y{0})
                {
                    max_val = block_max_val;
                    sum_val = block_sum_val;
                    if (overwrite)
                    {
                        *chunk_at<T>(dst, 2 * dst_offset) = R::from(max_val);
                        *chunk_at<T>(dst, 2 * dst_offset + 1) =
                            R::from(sum_val);
                    }
                    else
                    {
                        Y max_output;
                        Y sum_output =
                            R::to(*chunk_at<T>(dst, 2 * dst_offset + 1));
                        if (sum_output == zero)
                        {
                            max_output = max_val;
                            sum_output = sum_val;
                        }
                        else
                        {
                            max_output =
                                R::to(*chunk_at<T>(dst, 2 * dst_offset));
                            if (max_val < max_output)
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
                        *chunk_at<T>(dst, 2 * dst_offset) =
                            R::from(max_output);
                        *chunk_at<T>(dst, 2 * dst_offset + 1) =
                            R::from(sum_output);
                    }
                }
                else if (overwrite)
                {
                    *chunk_at<T>(dst, 2 * dst_offset) = R::from(zero);
                    *chunk_at<T>(dst, 2 * dst_offset + 1) = R::from(zero);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// maxsumexp, m == 1 path: the stock block-per-output reduction with the
// same BLOCK_ROW/LOOP geometry; only the source reads go through the
// chunk table.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_ROW, int LOOP>
__global__ void maxsumexp_m1_kernel(unsigned long long n,
    unsigned long long k, ChunkTable src, float beta_f, ChunkTable dst)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    long long const kk = static_cast<long long>(k);
    long long const src_l_block_end = (kk / BLOCK_ROW) * BLOCK_ROW;
    Y const one{1.0};
    constexpr int BLOCK_ROW_STEP = BLOCK_ROW / LOOP;
    bool const overwrite = (static_cast<Y>(beta_f) == Y{0.0});
    T src_val[LOOP];
    volatile __shared__ Y dst_block_max[BLOCK_ROW_STEP];
    volatile __shared__ Y dst_block_sumexp[BLOCK_ROW_STEP];
    Y dst_max = -INFINITY, dst_sumexp = 0.0;
    unsigned long long const out = blockIdx.x;
    for (long long src_l = threadIdx.x; src_l < src_l_block_end;
        src_l += BLOCK_ROW)
    {
        for (int c = 0; c < LOOP; ++c)
        {
            unsigned long long const flat =
                static_cast<unsigned long long>(src_l)
                + static_cast<unsigned long long>(c * BLOCK_ROW_STEP)
                + out * k;
            src_val[c] = *chunk_at<T>(src, flat);
        }
        for (int c = 0; c < LOOP; ++c)
        {
            Y const val = R::to(src_val[c]);
            if (!::isinf(val))
            {
                if (dst_max < val)
                {
                    dst_sumexp = one + ::exp(dst_max - val) * dst_sumexp;
                    dst_max = val;
                }
                else
                {
                    dst_sumexp += ::exp(val - dst_max);
                }
            }
        }
    }
    long long const src_l =
        static_cast<long long>(threadIdx.x) + src_l_block_end;
    int const c_max = static_cast<int>(
        (kk - src_l + BLOCK_ROW_STEP - 1) / BLOCK_ROW_STEP);
    for (int c = 0; c < c_max; ++c)
    {
        unsigned long long const flat = static_cast<unsigned long long>(src_l)
            + static_cast<unsigned long long>(c * BLOCK_ROW_STEP) + out * k;
        src_val[c] = *chunk_at<T>(src, flat);
    }
    for (int c = 0; c < c_max; ++c)
    {
        Y const val = R::to(src_val[c]);
        if (!::isinf(val))
        {
            if (dst_max < val)
            {
                dst_sumexp = one + ::exp(dst_max - val) * dst_sumexp;
                dst_max = val;
            }
            else
            {
                dst_sumexp += ::exp(val - dst_max);
            }
        }
    }
    dst_block_max[threadIdx.x] = dst_max;
    dst_block_sumexp[threadIdx.x] = dst_sumexp;
    __syncthreads();
    for (int c = BLOCK_ROW_STEP >> 1; c > 32; c >>= 1)
    {
        if (threadIdx.x < c)
        {
            Y max2 = dst_block_max[threadIdx.x + c];
            if (!::isinf(max2))
            {
                volatile Y &max = dst_block_max[threadIdx.x];
                volatile Y &sumexp = dst_block_sumexp[threadIdx.x];
                Y sumexp2 = dst_block_sumexp[threadIdx.x + c];
                if (max < max2)
                {
                    sumexp = sumexp2 + ::exp(max - max2) * sumexp;
                    max = max2;
                }
                else
                {
                    sumexp += ::exp(max2 - max) * sumexp2;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
    {
        for (int c = 32; c > 0; c >>= 1)
        {
            Y max2 = dst_block_max[threadIdx.x + c];
            if (!::isinf(max2))
            {
                volatile Y &max = dst_block_max[threadIdx.x];
                volatile Y &sumexp = dst_block_sumexp[threadIdx.x];
                Y sumexp2 = dst_block_sumexp[threadIdx.x + c];
                if (max < max2)
                {
                    sumexp = sumexp2 + ::exp(max - max2) * sumexp;
                    max = max2;
                }
                else
                {
                    sumexp += ::exp(max2 - max) * sumexp2;
                }
            }
        }
    }
    if (threadIdx.x == 0)
    {
        Y max2 = dst_block_max[0];
        Y sumexp2 = dst_block_sumexp[0];
        if (!::isinf(max2))
        {
            if (overwrite)
            {
                *chunk_at<T>(dst, 2 * out) = R::from(max2);
                *chunk_at<T>(dst, 2 * out + 1) = R::from(sumexp2);
            }
            else
            {
                Y max = R::to(*chunk_at<T>(dst, 2 * out));
                Y sumexp = R::to(*chunk_at<T>(dst, 2 * out + 1));
                if (sumexp == Y{0.0})
                {
                    sumexp = sumexp2;
                    max = max2;
                }
                else if (max < max2)
                {
                    sumexp = sumexp2 + ::exp(max - max2) * sumexp;
                    max = max2;
                }
                else
                {
                    sumexp += ::exp(max2 - max) * sumexp2;
                }
                *chunk_at<T>(dst, 2 * out) = R::from(max);
                *chunk_at<T>(dst, 2 * out + 1) = R::from(sumexp);
            }
        }
        else if (overwrite)
        {
            *chunk_at<T>(dst, 2 * out) = R::from(Y{0.0});
            *chunk_at<T>(dst, 2 * out + 1) = R::from(Y{0.0});
        }
    }
}

// ---------------------------------------------------------------------------
// swap_two_axes: [d0,d1,d2,d3,d4] -> [d0,d3,d2,d1,d4]
//
// Destination flat position p decodes with the destination axes
// (d0,d3,d2,d1,d4); the source flat position of the same logical cell is
// i0*d1*d2*d3*d4 + i1*d2*d3*d4 + i2*d3*d4 + i3*d4 + i4. Both positions
// go through their chunk table with the shift/mask map.
// ---------------------------------------------------------------------------

constexpr int kSwapWarps = 8;

//! d4 > 1 path: one warp per contiguous i4 fiber (coalesced both sides).
template <typename T>
__global__ void swap_fiber_kernel(unsigned long long d0,
    unsigned long long d1, unsigned long long d2, unsigned long long d3,
    unsigned long long d4, unsigned long long fiber_base,
    unsigned long long n_fibers, ChunkTable src, ChunkTable dst)
{
    unsigned long long const fiber = fiber_base
        + static_cast<unsigned long long>(blockIdx.x) * kSwapWarps
        + threadIdx.y;
    if (fiber >= fiber_base + n_fibers)
    {
        return;
    }
    unsigned long long rem = fiber;
    unsigned long long const i3 = rem % d3;
    rem /= d3;
    unsigned long long const i2 = rem % d2;
    rem /= d2;
    unsigned long long const i1 = rem % d1;
    unsigned long long const i0 = rem / d1;
    unsigned long long const src_base =
        ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4);
    unsigned long long const dst_base =
        ((((i0 * d3 + i3) * d2 + i2) * d1 + i1) * d4);
    for (unsigned long long i4 = threadIdx.x; i4 < d4; i4 += 32)
    {
        *chunk_at<T>(dst, dst_base + i4) = *chunk_at<T>(src,
            src_base + i4);
    }
}

//! d4 == 1 (and the split fallback): dst-driven flat walk.
template <typename T>
__global__ void swap_flat_kernel(unsigned long long d0,
    unsigned long long d1, unsigned long long d2, unsigned long long d3,
    unsigned long long d4, unsigned long long p_base,
    unsigned long long p_end, ChunkTable src, ChunkTable dst)
{
    unsigned long long const p = p_base
        + static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= p_end)
    {
        return;
    }
    unsigned long long const s0 = d3 * d2 * d1 * d4;
    unsigned long long const s3 = d2 * d1 * d4;
    unsigned long long const s2 = d1 * d4;
    unsigned long long const s1 = d4;
    unsigned long long r = p;
    unsigned long long const i0 = r / s0;
    r -= i0 * s0;
    unsigned long long const i3 = r / s3;
    r -= i3 * s3;
    unsigned long long const i2 = r / s2;
    r -= i2 * s2;
    unsigned long long const i1 = r / s1;
    unsigned long long const i4 = r - i1 * s1;
    unsigned long long const src_flat =
        ((((i0 * d1 + i1) * d2 + i2) * d3 + i3) * d4) + i4;
    *chunk_at<T>(dst, p) = *chunk_at<T>(src, src_flat);
}

// ---------------------------------------------------------------------------
// transpose: dst[i*n+j] = alpha * src[i+j*m], tiled so the destination
// writes and the source reads are both coalesced.
// ---------------------------------------------------------------------------

constexpr int kTransTile = 32;

template <typename T>
__global__ void transpose_tile_kernel(unsigned long long m,
    unsigned long long n, float alpha_f, unsigned long long i_base,
    unsigned long long j_base, ChunkTable src, ChunkTable dst)
{
    __shared__ T tile[kTransTile][kTransTile + 1];
    using R = Repr<T>;
    using Y = typename R::Y;
    Y const alpha = static_cast<Y>(alpha_f);
    unsigned long long const i0 = i_base
        + static_cast<unsigned long long>(blockIdx.y) * kTransTile;
    unsigned long long const j0 = j_base
        + static_cast<unsigned long long>(blockIdx.x) * kTransTile;
    // Load phase: consecutive tx address consecutive i (contiguous src).
    unsigned long long const i_load = i0 + threadIdx.x;
    unsigned long long const j_load = j0 + threadIdx.y;
    if (i_load < m && j_load < n)
    {
        T const v = *chunk_at<T>(src, i_load + j_load * m);
        tile[threadIdx.y][threadIdx.x] =
            R::from(alpha * R::to(v));
    }
    __syncthreads();
    // Store phase: consecutive tx address consecutive j (contiguous dst).
    unsigned long long const i_store = i0 + threadIdx.y;
    unsigned long long const j_store = j0 + threadIdx.x;
    if (i_store < m && j_store < n)
    {
        *chunk_at<T>(dst, i_store * n + j_store) =
            tile[threadIdx.x][threadIdx.y];
    }
}

// ---------------------------------------------------------------------------
// conv2d family: the stock kernels' exact arithmetic with chunk tables
//
// Each kernel thread owns one destination flat position p (the stock
// decode: i + dst_m*(j + dst_n*(c + channels*b))), writes it through the
// destination table, and reads every src1/src2 element through the
// source tables. The host splits the destination flat range until the
// touched chunk span of every operand fits the by-value table.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void conv2d_inplace_chunk_kernel(long long src1_m,
    long long src1_n, long long src1_channels, long long batch,
    long long src2_m, long long src2_n, long long dilation_m,
    long long dilation_n, long long dst_channels, long long offset_m,
    long long offset_n, float alpha_f, long long dst_m, long long dst_n,
    long long stride_m, long long stride_n, float beta_f,
    unsigned long long p_base, unsigned long long p_end,
    ChunkTable src1t, ChunkTable src2t, ChunkTable dstt)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const p = p_base
        + static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= p_end)
    {
        return;
    }
    unsigned long long r = p;
    long long const dst_i = static_cast<long long>(r % dst_m);
    r /= static_cast<unsigned long long>(dst_m);
    long long const dst_j = static_cast<long long>(
        r % static_cast<unsigned long long>(dst_n));
    r /= static_cast<unsigned long long>(dst_n);
    long long const oc = static_cast<long long>(
        r % static_cast<unsigned long long>(dst_channels));
    long long const b = static_cast<long long>(
        r / static_cast<unsigned long long>(dst_channels));
    long long const dst_start_m = ::max(
        (offset_m - dilation_m * (src2_m - 1) + stride_m - 1) / stride_m,
        0LL);
    long long const dst_end_m = ::min(
        (offset_m + src1_m + stride_m - 1) / stride_m, dst_m);
    long long const dst_start_n = ::max(
        (offset_n - dilation_n * (src2_n - 1) + stride_n - 1) / stride_n,
        0LL);
    long long const dst_end_n = ::min(
        (offset_n + src1_n + stride_n - 1) / stride_n, dst_n);
    T *dst_val = chunk_at<T>(dstt, p);
    if (dst_i >= dst_start_m and dst_i < dst_end_m
        and dst_j >= dst_start_n and dst_j < dst_end_n)
    {
        long long const src1_step = src1_n * src1_m;
        long long const src2_ic_step = src2_n * src2_m;
        long long const src2_oc_step = src2_ic_step * src1_channels;
        Y conv{0.0}, c{0}, y, t;
        long long src1_start_m = dst_i * stride_m - offset_m;
        if (src1_start_m < 0)
        {
            long long const neg_rem = (-src1_start_m) % dilation_m;
            src1_start_m = neg_rem == 0 ? 0 : dilation_m - neg_rem;
        }
        long long const src1_end_m = ::min(
            dst_i * stride_m - offset_m + dilation_m * (src2_m - 1) + 1,
            src1_m);
        long long src1_start_n = dst_j * stride_n - offset_n;
        if (src1_start_n < 0)
        {
            long long const neg_rem = (-src1_start_n) % dilation_n;
            src1_start_n = neg_rem == 0 ? 0 : dilation_n - neg_rem;
        }
        long long const src1_end_n = ::min(
            dst_j * stride_n - offset_n + dilation_n * (src2_n - 1) + 1,
            src1_n);
        for (long long src1_i = src1_start_m; src1_i < src1_end_m;
            src1_i += dilation_m)
        {
            for (long long src1_j = src1_start_n; src1_j < src1_end_n;
                src1_j += dilation_n)
            {
                unsigned long long const s1_base =
                    static_cast<unsigned long long>(src1_i
                        + (b * src1_channels * src1_n + src1_j) * src1_m);
                unsigned long long const s2_base =
                    static_cast<unsigned long long>(
                        (src1_i + offset_m - dst_i * stride_m) / dilation_m
                        + (src1_j + offset_n - dst_j * stride_n)
                            / dilation_n * src2_m
                        + oc * src2_oc_step);
                for (long long ic = 0; ic < src1_channels; ++ic)
                {
                    Y const src1_val{R::to(*chunk_at<T>(src1t,
                        s1_base + static_cast<unsigned long long>(
                            src1_step * ic)))};
                    Y const src2_val{R::to(*chunk_at<T>(src2t,
                        s2_base + static_cast<unsigned long long>(
                            src2_ic_step * ic)))};
                    y = src1_val * src2_val - c;
                    t = conv + y;
                    c = (t - conv) - y;
                    conv = t;
                }
            }
        }
        if (beta_f == 0.0F)
        {
            *dst_val = R::from(static_cast<Y>(alpha_f) * conv);
        }
        else
        {
            Y const old{R::to(*dst_val)};
            *dst_val = R::from((static_cast<Y>(beta_f) * old
                - static_cast<Y>(alpha_f) * c)
                + static_cast<Y>(alpha_f) * conv);
        }
    }
    else
    {
        if (beta_f == 0.0F)
        {
            *dst_val = R::from(Y{0.0});
        }
        else if (beta_f != 1.0F)
        {
            Y const old{R::to(*dst_val)};
            *dst_val = R::from(static_cast<Y>(beta_f) * old);
        }
    }
}

template <typename T>
__global__ void conv2d_bwd_input_inplace_chunk_kernel(long long src1_m,
    long long src1_n, long long stride_m, long long stride_n,
    long long src1_channels, long long batch, long long src2_m,
    long long src2_n, long long dilation_m, long long dilation_n,
    long long dst_channels, long long offset_m, long long offset_n,
    float alpha_f, long long dst_m, long long dst_n, float beta_f,
    unsigned long long p_base, unsigned long long p_end,
    ChunkTable src1t, ChunkTable src2t, ChunkTable dstt)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const p = p_base
        + static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= p_end)
    {
        return;
    }
    unsigned long long r = p;
    long long const dst_i = static_cast<long long>(r % dst_m);
    r /= static_cast<unsigned long long>(dst_m);
    long long const dst_j = static_cast<long long>(
        r % static_cast<unsigned long long>(dst_n));
    r /= static_cast<unsigned long long>(dst_n);
    long long const oc = static_cast<long long>(
        r % static_cast<unsigned long long>(dst_channels));
    long long const b = static_cast<long long>(
        r / static_cast<unsigned long long>(dst_channels));
    long long const dst_start_m = ::max(-offset_m, 0LL);
    long long const dst_end_m = ::min(
        dilation_m * (src2_m - 1) + stride_m * (src1_m - 1) - offset_m + 1,
        dst_m);
    long long const dst_start_n = ::max(-offset_n, 0LL);
    long long const dst_end_n = ::min(
        dilation_n * (src2_n - 1) + stride_n * (src1_n - 1) - offset_n + 1,
        dst_n);
    T *dst_val = chunk_at<T>(dstt, p);
    if (dst_i >= dst_start_m and dst_i < dst_end_m
        and dst_j >= dst_start_n and dst_j < dst_end_n)
    {
        long long const src1_step = src1_n * src1_m;
        long long const src2_oc_step = src2_n * src2_m;
        long long const src2_ic_step = src2_oc_step * dst_channels;
        Y conv{0.0}, c{0}, y, t;
        long long src1_start_m =
            (dst_i + offset_m - dilation_m * (src2_m - 1) + stride_m - 1)
            / stride_m;
        src1_start_m = ::max(src1_start_m, 0LL);
        long long const src1_end_m = ::min(
            (dst_i + offset_m + stride_m) / stride_m, src1_m);
        long long src1_start_n =
            (dst_j + offset_n - dilation_n * (src2_n - 1) + stride_n - 1)
            / stride_n;
        src1_start_n = ::max(src1_start_n, 0LL);
        long long const src1_end_n = ::min(
            (dst_j + offset_n + stride_n) / stride_n, src1_n);
        for (long long src1_i = src1_start_m; src1_i < src1_end_m;
            ++src1_i)
        {
            long long const tmp_i = dst_i + offset_m - stride_m * src1_i;
            if (tmp_i % dilation_m != 0)
            {
                continue;
            }
            for (long long src1_j = src1_start_n; src1_j < src1_end_n;
                ++src1_j)
            {
                long long const tmp_j =
                    dst_j + offset_n - stride_n * src1_j;
                if (tmp_j % dilation_n != 0)
                {
                    continue;
                }
                unsigned long long const s1_base =
                    static_cast<unsigned long long>(src1_i
                        + (b * src1_channels * src1_n + src1_j) * src1_m);
                unsigned long long const s2_base =
                    static_cast<unsigned long long>(
                        tmp_i / dilation_m
                        + tmp_j / dilation_n * src2_m
                        + oc * src2_oc_step);
                for (long long ic = 0; ic < src1_channels; ++ic)
                {
                    Y const src1_val{R::to(*chunk_at<T>(src1t,
                        s1_base + static_cast<unsigned long long>(
                            src1_step * ic)))};
                    Y const src2_val{R::to(*chunk_at<T>(src2t,
                        s2_base + static_cast<unsigned long long>(
                            src2_ic_step * ic)))};
                    y = src1_val * src2_val - c;
                    t = conv + y;
                    c = (t - conv) - y;
                    conv = t;
                }
            }
        }
        if (beta_f == 0.0F)
        {
            *dst_val = R::from(static_cast<Y>(alpha_f) * conv);
        }
        else
        {
            Y const old{R::to(*dst_val)};
            *dst_val = R::from((static_cast<Y>(beta_f) * old
                - static_cast<Y>(alpha_f) * c)
                + static_cast<Y>(alpha_f) * conv);
        }
    }
    else
    {
        if (beta_f == 0.0F)
        {
            *dst_val = R::from(Y{0.0});
        }
        else if (beta_f != 1.0F)
        {
            Y const old{R::to(*dst_val)};
            *dst_val = R::from(static_cast<Y>(beta_f) * old);
        }
    }
}

template <typename T>
__global__ void conv2d_bwd_weight_inplace_chunk_kernel(long long src1_m,
    long long src1_n, long long src1_channels, long long batch,
    long long src2_m, long long src2_n, long long stride_m,
    long long stride_n, long long src2_channels, long long offset_m,
    long long offset_n, float alpha_f, long long dst_m, long long dst_n,
    long long dilation_m, long long dilation_n, float beta_f,
    unsigned long long p_base, unsigned long long p_end,
    ChunkTable src1t, ChunkTable src2t, ChunkTable dstt)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const p = p_base
        + static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= p_end)
    {
        return;
    }
    unsigned long long r = p;
    long long const dst_i = static_cast<long long>(r % dst_m);
    r /= static_cast<unsigned long long>(dst_m);
    long long const dst_j = static_cast<long long>(
        r % static_cast<unsigned long long>(dst_n));
    r /= static_cast<unsigned long long>(dst_n);
    long long const ic = static_cast<long long>(
        r % static_cast<unsigned long long>(src1_channels));
    long long const oc = static_cast<long long>(
        r / static_cast<unsigned long long>(src1_channels));
    long long const dst_start_m = ::max(
        (offset_m - stride_m * (src2_m - 1) + dilation_m - 1) / dilation_m,
        0LL);
    long long const dst_end_m = ::min(
        (src1_m - 1 + offset_m + dilation_m) / dilation_m, dst_m);
    long long const dst_start_n = ::max(
        (offset_n - stride_n * (src2_n - 1) + dilation_n - 1) / dilation_n,
        0LL);
    long long const dst_end_n = ::min(
        (src1_n - 1 + offset_n + dilation_n) / dilation_n, dst_n);
    T *dst_val = chunk_at<T>(dstt, p);
    if (dst_i >= dst_start_m and dst_i < dst_end_m
        and dst_j >= dst_start_n and dst_j < dst_end_n)
    {
        long long const src1_step = src1_channels * src1_n * src1_m;
        long long const src2_step = src2_channels * src2_n * src2_m;
        Y conv{0.0}, c{0}, y, t;
        long long src1_start_m = dilation_m * dst_i - offset_m;
        if (src1_start_m < 0)
        {
            long long const neg_rem = (-src1_start_m) % stride_m;
            src1_start_m = neg_rem == 0 ? 0 : stride_m - neg_rem;
        }
        long long const src1_end_m = ::min(
            dilation_m * dst_i - offset_m + stride_m * (src2_m - 1) + 1,
            src1_m);
        long long src1_start_n = dilation_n * dst_j - offset_n;
        if (src1_start_n < 0)
        {
            long long const neg_rem = (-src1_start_n) % stride_n;
            src1_start_n = neg_rem == 0 ? 0 : stride_n - neg_rem;
        }
        long long const src1_end_n = ::min(
            dilation_n * dst_j - offset_n + stride_n * (src2_n - 1) + 1,
            src1_n);
        for (long long src1_i = src1_start_m; src1_i < src1_end_m;
            src1_i += stride_m)
        {
            for (long long src1_j = src1_start_n; src1_j < src1_end_n;
                src1_j += stride_n)
            {
                unsigned long long const s1_base =
                    static_cast<unsigned long long>(src1_i
                        + (ic * src1_n + src1_j) * src1_m);
                unsigned long long const s2_base =
                    static_cast<unsigned long long>(
                        (src1_i + offset_m - dilation_m * dst_i) / stride_m
                        + (oc * src2_n
                            + (src1_j + offset_n - dilation_n * dst_j)
                                / stride_n) * src2_m);
                for (long long b = 0; b < batch; ++b)
                {
                    Y const src1_val{R::to(*chunk_at<T>(src1t,
                        s1_base + static_cast<unsigned long long>(
                            src1_step * b)))};
                    Y const src2_val{R::to(*chunk_at<T>(src2t,
                        s2_base + static_cast<unsigned long long>(
                            src2_step * b)))};
                    y = src1_val * src2_val - c;
                    t = conv + y;
                    c = (t - conv) - y;
                    conv = t;
                }
            }
        }
        if (beta_f == 0.0F)
        {
            *dst_val = R::from(static_cast<Y>(alpha_f) * conv);
        }
        else
        {
            Y const old{R::to(*dst_val)};
            *dst_val = R::from((static_cast<Y>(beta_f) * old
                - static_cast<Y>(alpha_f) * c)
                + static_cast<Y>(alpha_f) * conv);
        }
    }
    else
    {
        if (beta_f == 0.0F)
        {
            *dst_val = R::from(Y{0.0});
        }
        else if (beta_f != 1.0F)
        {
            Y const old{R::to(*dst_val)};
            *dst_val = R::from(static_cast<Y>(beta_f) * old);
        }
    }
}

// ---------------------------------------------------------------------------
// randn: per-position chameleon value (seed jumped by the absolute
// underlying position), strided output offset through the chunk table.
// ---------------------------------------------------------------------------

constexpr int kRandnMaxNdim = 8;

struct RandnChunkGeom
{
    long long ndim;
    unsigned long long seed;
    float mean;
    float stddev;
    long long start[kRandnMaxNdim];
    long long shape[kRandnMaxNdim];
    long long stride[kRandnMaxNdim];
    long long underlying_shape[kRandnMaxNdim];
};

__device__ __forceinline__ unsigned long long rnd64_jump_chunk(
    unsigned long long n, unsigned long long seed)
{
    constexpr unsigned long long a = 6364136223846793005ULL;
    constexpr unsigned long long c = 1ULL;
    unsigned long long a_k = a;
    unsigned long long c_k = c;
    unsigned long long ran = seed;
    n <<= 1;
    for (int i = 0; n; n >>= 1, ++i)
    {
        if (n & 1)
        {
            ran = a_k * ran + c_k;
        }
        c_k *= (a_k + 1);
        a_k *= a_k;
    }
    return ran;
}

__device__ __forceinline__ float slaran_chunk(unsigned long long &ran)
{
    float const value = ran * 5.4210108624275222e-20F;
    ran = 6364136223846793005ULL * ran + 1ULL;
    return value;
}

__device__ __forceinline__ double dlaran_chunk(unsigned long long &ran)
{
    double const value = ran * 5.4210108624275222e-20;
    ran = 6364136223846793005ULL * ran + 1ULL;
    return value;
}

template <typename Y>
__device__ __forceinline__ Y chameleon_randn_chunk(unsigned long long &seed,
    Y mean, Y stddev)
{
    constexpr Y two = Y{2.0};
    constexpr Y twopi = Y{6.2831853071795864769252867663};
    if constexpr (std::is_same_v<Y, double>)
    {
        Y const t1 = dlaran_chunk(seed);
        Y const t2 = dlaran_chunk(seed) * twopi;
        Y const t3 = sqrt(-two * log(t1)) * cos(t2);
        return stddev * t3 + mean;
    }
    else
    {
        Y const t1 = slaran_chunk(seed);
        Y const t2 = slaran_chunk(seed) * twopi;
        Y const t3 = sqrtf(-two * logf(t1)) * cosf(t2);
        return stddev * t3 + mean;
    }
}

template <typename T>
__global__ void randn_chunk_kernel(unsigned long long p_base,
    unsigned long long p_end, unsigned long long seed, float mean_f,
    float stddev_f, RandnChunkGeom geom, ChunkTable dstt)
{
    using R = Repr<T>;
    using Y = typename R::Y;
    unsigned long long const p = p_base
        + static_cast<unsigned long long>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (p >= p_end)
    {
        return;
    }
    Y const mean{mean_f};
    Y const stddev{stddev_f};
    long long lin = static_cast<long long>(p);
    long long offset = 0;
    long long under = 0;
    long long under_stride = 1;
    for (long long d = 0; d < geom.ndim; ++d)
    {
        long long const coord = lin % geom.shape[d];
        lin /= geom.shape[d];
        offset += coord * geom.stride[d];
        under += (coord + geom.start[d]) * under_stride;
        under_stride *= geom.underlying_shape[d];
    }
    unsigned long long local = rnd64_jump_chunk(
        static_cast<unsigned long long>(under), seed);
    Y const value = chameleon_randn_chunk<Y>(local, mean, stddev);
    *chunk_at<T>(dstt, static_cast<unsigned long long>(offset)) =
        R::from(value);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

constexpr unsigned long long kUnchunkedShift = 63;

//! Elements in a full chunk (the whole buffer when unchunked).
inline unsigned long long chunk_elems_of(Buffer const &buf,
    std::size_t elem_size, char const *who)
{
    if (buf.chunk_count() > 1)
    {
        if (buf.chunk_bytes(0) != static_cast<std::size_t>(kChunkClassBytes))
        {
            std::fprintf(stderr,
                "[nnhaul] %s: chunk 0 is %zu bytes, not the compile-time "
                "class %zu\n",
                who, buf.chunk_bytes(0),
                static_cast<std::size_t>(kChunkClassBytes));
            std::abort();
        }
        unsigned long long const elems =
            static_cast<unsigned long long>(kChunkClassBytes) / elem_size;
        if (elem_size == 0 || (elem_size & (elem_size - 1)) != 0
            || (elems & (elems - 1)) != 0)
        {
            std::fprintf(stderr,
                "[nnhaul] %s: element size %zu is not a power of two\n",
                who, elem_size);
            std::abort();
        }
        return elems;
    }
    unsigned long long const total = buf.nbytes() / elem_size;
    return total > 0 ? total : 1;
}

//! Build a table for the global chunk range [chunk_lo, chunk_hi)
//! (hi exclusive). An unchunked buffer is the single chunk 0.
ChunkTable make_table(Buffer const &buf, std::size_t elem_size,
    std::size_t chunk_lo, std::size_t chunk_hi, char const *who)
{
    ChunkTable t{};
    std::size_t const nch = buf.chunk_count();
    if (chunk_lo > chunk_hi || chunk_hi > nch
        || chunk_hi - chunk_lo > static_cast<std::size_t>(kMaxTableChunks))
    {
        std::fprintf(stderr,
            "[nnhaul] %s: chunk range [%zu, %zu) is invalid for %zu "
            "chunks (by-value table %d)\n",
            who, chunk_lo, chunk_hi, nch, kMaxTableChunks);
        std::abort();
    }
    t.n = static_cast<int>(chunk_hi - chunk_lo);
    t.base = static_cast<int>(chunk_lo);
    if (nch > 1)
    {
        unsigned long long const elems = chunk_elems_of(buf, elem_size,
            who);
        t.shift = static_cast<unsigned long long>(log2_u64(elems));
        t.mask = elems - 1;
    }
    else
    {
        // c = pos >> 63 == 0, off = pos & ~0ull == pos.
        t.shift = kUnchunkedShift;
        t.mask = ~0ull;
    }
    for (std::size_t c = chunk_lo; c < chunk_hi; ++c)
    {
        t.ptr[c - chunk_lo] = buf.get_chunk_ptr<char>(c);
    }
    return t;
}

ChunkTable table_of(Buffer const &buf, std::size_t elem_size,
    char const *who)
{
    return make_table(buf, elem_size, 0, buf.chunk_count(), who);
}

//! Global chunk id of a flat position (0 for an unchunked buffer).
inline unsigned long long chunk_id_of(Buffer const &buf,
    std::size_t elem_size, unsigned long long pos, char const *who)
{
    if (buf.chunk_count() <= 1)
    {
        return 0;
    }
    return pos / chunk_elems_of(buf, elem_size, who);
}

inline void require_kernel(char const *who)
{
    (void)who;
    (void)::nnhaul::cuda_stream();
}

inline void check_launch(char const *who)
{
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[nnhaul] %s launch failed: %s\n", who,
            cudaGetErrorString(err));
        std::abort();
    }
}

template <class T> struct TypeTag
{
    using type = T;
};

template <class F> void dispatch_type(ElemType t, F &&f)
{
    switch (t)
    {
    case ElemType::F32:
        f(TypeTag<float>{});
        break;
    case ElemType::F64:
        f(TypeTag<double>{});
        break;
    case ElemType::BF16:
        f(TypeTag<__nv_bfloat16>{});
        break;
    case ElemType::F16:
        f(TypeTag<__half>{});
        break;
    }
}

//! Conservative source-position bounds of a contiguous destination
//! position range under the swap map. Each destination digit's range is
//! the endpoint range, widened to the full extent when the interval
//! length covers at least one full period of that digit (a superset of
//! the exact set); the source flat map is monotone in each axis, so the
//! corner combinations bound every source position in the range.
inline void swap_src_bounds(unsigned long long d0, unsigned long long d1,
    unsigned long long d2, unsigned long long d3, unsigned long long d4,
    unsigned long long p0, unsigned long long p1,
    unsigned long long &src_min, unsigned long long &src_max)
{
    unsigned long long const dims[5] = {d0, d3, d2, d1, d4};
    unsigned long long const strides[5] = {d3 * d2 * d1 * d4,
        d2 * d1 * d4, d1 * d4, d4, 1};
    unsigned long long lo[5];
    unsigned long long hi[5];
    unsigned long long const len = p1 - p0;
    for (int k = 0; k < 5; ++k)
    {
        unsigned long long const stride = strides[k];
        unsigned long long const dim = dims[k];
        lo[k] = (p0 / stride) % dim;
        hi[k] = ((p1 - 1) / stride) % dim;
        if (len >= stride * dim)
        {
            lo[k] = 0;
            hi[k] = dim - 1;
        }
    }
    // Destination digits are (i0, i3, i2, i1, i4).
    unsigned long long const c0 = d1 * d2 * d3 * d4;
    unsigned long long const c1 = d2 * d3 * d4;
    unsigned long long const c2 = d3 * d4;
    unsigned long long const c3 = d4;
    src_min = lo[0] * c0 + lo[3] * c1 + lo[2] * c2 + lo[1] * c3
        + lo[4];
    src_max = hi[0] * c0 + hi[3] * c1 + hi[2] * c2 + hi[1] * c3
        + hi[4];
}

template <typename T>
void swap_emit_range(unsigned long long d0, unsigned long long d1,
    unsigned long long d2, unsigned long long d3, unsigned long long d4,
    std::size_t elem_size, unsigned long long p0, unsigned long long p1,
    Buffer const &src, Buffer &dst, char const *who)
{
    if (p0 >= p1)
    {
        return;
    }
    unsigned long long src_min = 0;
    unsigned long long src_max = 0;
    swap_src_bounds(d0, d1, d2, d3, d4, p0, p1, src_min, src_max);
    unsigned long long const cd0 = chunk_id_of(dst, elem_size, p0, who);
    unsigned long long const cd1 = chunk_id_of(dst, elem_size, p1 - 1,
        who);
    unsigned long long const cs0 = chunk_id_of(src, elem_size, src_min,
        who);
    unsigned long long const cs1 = chunk_id_of(src, elem_size, src_max,
        who);
    if (cd1 - cd0 < static_cast<unsigned long long>(kMaxTableChunks)
        && cs1 - cs0 < static_cast<unsigned long long>(kMaxTableChunks))
    {
        ChunkTable const st = make_table(src, elem_size, cs0, cs1 + 1, who);
        ChunkTable const dt = make_table(dst, elem_size, cd0, cd1 + 1, who);
        unsigned long long const blocks = (p1 - p0 + 255) / 256;
        swap_flat_kernel<T><<<static_cast<unsigned>(blocks), 256, 0,
            ::nnhaul::cuda_stream()>>>(d0, d1, d2, d3, d4, p0, p1, st, dt);
        return;
    }
    if (p1 - p0 == 1)
    {
        std::fprintf(stderr,
            "[nnhaul] %s: a single destination element spans more than "
            "%d source chunks; cannot build the by-value table\n",
            who, kMaxTableChunks);
        std::abort();
    }
    unsigned long long const mid = p0 + (p1 - p0) / 2;
    swap_emit_range<T>(d0, d1, d2, d3, d4, elem_size, p0, mid, src, dst,
        who);
    swap_emit_range<T>(d0, d1, d2, d3, d4, elem_size, mid, p1, src, dst,
        who);
}

template <typename T>
void transpose_emit_tiles(unsigned long long m, unsigned long long n,
    float alpha, std::size_t elem_size, unsigned long long it0,
    unsigned long long it1, unsigned long long jt0,
    unsigned long long jt1, Buffer const &src, Buffer &dst,
    char const *who)
{
    if (it0 >= it1 || jt0 >= jt1 || it0 * kTransTile >= m
        || jt0 * kTransTile >= n)
    {
        return;
    }
    unsigned long long const i_lo = it0 * kTransTile;
    unsigned long long const i_hi_tmp = it1 * kTransTile;
    unsigned long long const i_hi = (i_hi_tmp < m ? i_hi_tmp : m) - 1;
    unsigned long long const j_lo = jt0 * kTransTile;
    unsigned long long const j_hi_tmp = jt1 * kTransTile;
    unsigned long long const j_hi = (j_hi_tmp < n ? j_hi_tmp : n) - 1;
    unsigned long long const src_min = i_lo + j_lo * m;
    unsigned long long const src_max = i_hi + j_hi * m;
    unsigned long long const dst_min = i_lo * n + j_lo;
    unsigned long long const dst_max = i_hi * n + j_hi;
    unsigned long long const cs0 = chunk_id_of(src, elem_size, src_min,
        who);
    unsigned long long const cs1 = chunk_id_of(src, elem_size, src_max,
        who);
    unsigned long long const cd0 = chunk_id_of(dst, elem_size, dst_min,
        who);
    unsigned long long const cd1 = chunk_id_of(dst, elem_size, dst_max,
        who);
    if (cs1 - cs0 < static_cast<unsigned long long>(kMaxTableChunks)
        && cd1 - cd0 < static_cast<unsigned long long>(kMaxTableChunks))
    {
        ChunkTable const st = make_table(src, elem_size, cs0, cs1 + 1, who);
        ChunkTable const dt = make_table(dst, elem_size, cd0, cd1 + 1, who);
        dim3 const blocks(static_cast<unsigned>(jt1 - jt0),
            static_cast<unsigned>(it1 - it0));
        dim3 const threads(kTransTile, kTransTile);
        transpose_tile_kernel<T><<<blocks, threads, 0, ::nnhaul::cuda_stream()>>>(
            m, n, alpha, i_lo, j_lo, st, dt);
        return;
    }
    if (it1 - it0 == 1 && jt1 - jt0 == 1)
    {
        std::fprintf(stderr,
            "[nnhaul] %s: one %dx%d tile spans more than %d chunks; "
            "cannot build the by-value table\n",
            who, kTransTile, kTransTile, kMaxTableChunks);
        std::abort();
    }
    if (jt1 - jt0 >= it1 - it0)
    {
        unsigned long long const mid = jt0 + (jt1 - jt0) / 2;
        transpose_emit_tiles<T>(m, n, alpha, elem_size, it0, it1, jt0,
            mid, src, dst, who);
        transpose_emit_tiles<T>(m, n, alpha, elem_size, it0, it1, mid,
            jt1, src, dst, who);
    }
    else
    {
        unsigned long long const mid = it0 + (it1 - it0) / 2;
        transpose_emit_tiles<T>(m, n, alpha, elem_size, it0, mid, jt0,
            jt1, src, dst, who);
        transpose_emit_tiles<T>(m, n, alpha, elem_size, mid, it1, jt0,
            jt1, src, dst, who);
    }
}

} // namespace

void chunk_multiply_inplace(Buffer const &src, Buffer &dst,
    std::size_t nelems, float alpha, ElemType type)
{
    require_kernel("chunk_multiply_inplace");
    if (nelems == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    ChunkTable const st = table_of(src, es, "multiply_inplace");
    ChunkTable const dt = table_of(dst, es, "multiply_inplace");
    unsigned long long const total = nelems;
    unsigned int const threads = 256;
    unsigned long long const blocks = (total + threads - 1) / threads;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        multiply_inplace_kernel<T><<<static_cast<unsigned>(blocks),
            threads, 0, ::nnhaul::cuda_stream()>>>(total, alpha, st, dt);
    });
    check_launch("multiply_inplace");
}

void chunk_sgd_step(Buffer const &grad, Buffer &velocity, Buffer &param,
    long long num_iter, std::size_t num_elems, float momentum, float lr,
    float weight_decay, float dampening, bool nesterov, ElemType type)
{
    require_kernel("chunk_sgd_step");
    if (num_elems == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    ChunkTable const gt = table_of(grad, es, "sgd_step");
    ChunkTable const vt = table_of(velocity, es, "sgd_step");
    ChunkTable const pt = table_of(param, es, "sgd_step");
    unsigned long long const total = num_elems;
    unsigned int const threads = 256;
    unsigned long long const blocks = (total + threads - 1) / threads;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        sgd_step_kernel<T><<<static_cast<unsigned>(blocks), threads, 0,
            ::nnhaul::cuda_stream()>>>(num_iter, total, momentum, lr, weight_decay,
            dampening, nesterov, gt, vt, pt);
    });
    check_launch("sgd_step");
}

void chunk_subtract_indexed_outputs(Buffer const &labels, Buffer &dst,
    std::size_t n_labels, std::size_t n_outputs, long long ignore_index,
    float val, ElemType type)
{
    require_kernel("chunk_subtract_indexed_outputs");
    if (n_labels == 0 || n_outputs == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    ChunkTable const lt = table_of(labels, sizeof(int64_t),
        "subtract_indexed_outputs");
    ChunkTable const dt = table_of(dst, es, "subtract_indexed_outputs");
    unsigned int const threads = 256;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        subtract_indexed_outputs_kernel<T><<<
            static_cast<unsigned>(n_outputs), threads, 0, ::nnhaul::cuda_stream()>>>(
            n_labels, n_outputs, ignore_index, val, lt, dt);
    });
    check_launch("subtract_indexed_outputs");
}

void chunk_sum_fiber(Buffer const &src, Buffer &dst, std::size_t m,
    std::size_t n, std::size_t k, std::size_t batch, float alpha,
    float beta, ElemType type)
{
    require_kernel("chunk_sum_fiber");
    if (m == 0 || k == 0 || batch == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    ChunkTable const st = table_of(src, es, "sum_fiber");
    ChunkTable const dt = table_of(dst, es, "sum_fiber");
    dim3 threads(1, static_cast<unsigned>(m < 32 ? m : 32),
        static_cast<unsigned>(n < 32 ? n : 32));
    unsigned long long const outs =
        static_cast<unsigned long long>(k) * batch;
    dim3 blocks(static_cast<unsigned>(outs), 1, 1);
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        sum_fiber_kernel<T><<<blocks, threads, 0, ::nnhaul::cuda_stream()>>>(m, n, k,
            batch, alpha, st, beta, dt);
    });
    check_launch("sum_fiber");
}

void chunk_total_sum_accum(Buffer const &logsumexp, Buffer const &src,
    Buffer const &labels, Buffer &val, float alpha, std::size_t n_labels,
    std::size_t n_outputs, long long ignore_index, ElemType type)
{
    require_kernel("chunk_total_sum_accum");
    if (n_labels == 0 || n_outputs == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    ChunkTable const lt = table_of(logsumexp, es, "total_sum_accum");
    ChunkTable const st = table_of(src, es, "total_sum_accum");
    ChunkTable const bt = table_of(labels, sizeof(int64_t),
        "total_sum_accum");
    ChunkTable const vt = table_of(val, sizeof(float), "total_sum_accum");
    unsigned int const threads = 256;
    unsigned long long const blocks = (n_outputs + threads - 1) / threads;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        total_sum_accum_kernel<T><<<static_cast<unsigned>(blocks), threads,
            0, ::nnhaul::cuda_stream()>>>(alpha, n_labels, n_outputs, ignore_index, lt,
            st, bt, vt);
    });
    check_launch("total_sum_accum");
}

void chunk_maxsumexp(Buffer const &src, Buffer &dst, std::size_t m,
    std::size_t n, std::size_t k, float beta, ElemType type)
{
    require_kernel("chunk_maxsumexp");
    if (m == 0 || n == 0 || k == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    ChunkTable const st = table_of(src, es, "maxsumexp");
    ChunkTable const dt = table_of(dst, es, "maxsumexp");
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (m == 1)
        {
            if (k <= 1024)
            {
                maxsumexp_m1_kernel<T, 1024, 16><<<static_cast<unsigned>(n),
                    64, 0, ::nnhaul::cuda_stream()>>>(n, k, st, beta, dt);
            }
            else if (k <= 2048)
            {
                maxsumexp_m1_kernel<T, 2048, 16><<<static_cast<unsigned>(n),
                    128, 0, ::nnhaul::cuda_stream()>>>(n, k, st, beta, dt);
            }
            else
            {
                maxsumexp_m1_kernel<T, 4096, 16><<<static_cast<unsigned>(n),
                    256, 0, ::nnhaul::cuda_stream()>>>(n, k, st, beta, dt);
            }
        }
        else
        {
            unsigned long long m_per_block = 1;
            unsigned long long n_per_block = 1;
            unsigned long long by = m;
            unsigned long long bz = n;
            if (m > 65535)
            {
                m_per_block = (m + 65534) / 65535;
                by = (m + m_per_block - 1) / m_per_block;
            }
            if (n > 65535)
            {
                n_per_block = (n + 65534) / 65535;
                bz = (n + n_per_block - 1) / n_per_block;
            }
            dim3 const blocks(1, static_cast<unsigned>(by),
                static_cast<unsigned>(bz));
            dim3 const threads(32, 1, 1);
            maxsumexp_kernel<T><<<blocks, threads, 0, ::nnhaul::cuda_stream()>>>(m,
                m_per_block, n, n_per_block, k, st, beta, dt);
        }
    });
    check_launch("maxsumexp");
}

void chunk_swap_two_axes(Buffer const &src, Buffer &dst, std::size_t d0,
    std::size_t d1, std::size_t d2, std::size_t d3, std::size_t d4,
    ElemType type)
{
    require_kernel("chunk_swap_two_axes");
    if (d0 == 0 || d1 == 0 || d2 == 0 || d3 == 0 || d4 == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    unsigned long long const ud0 = d0;
    unsigned long long const ud1 = d1;
    unsigned long long const ud2 = d2;
    unsigned long long const ud3 = d3;
    unsigned long long const ud4 = d4;
    unsigned long long const total = ud0 * ud1 * ud2 * ud3 * ud4;
    bool const fits = src.chunk_count()
            <= static_cast<std::size_t>(kMaxTableChunks)
        && dst.chunk_count() <= static_cast<std::size_t>(kMaxTableChunks);
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (!fits)
        {
            // Wide operands: contiguous destination ranges whose source
            // and destination chunk spans fit the table, split
            // recursively; never whole-buffer staging.
            swap_emit_range<T>(ud0, ud1, ud2, ud3, ud4, es, 0, total, src,
                dst, "swap_two_axes");
            return;
        }
        ChunkTable const st = table_of(src, es, "swap_two_axes");
        ChunkTable const dt = table_of(dst, es, "swap_two_axes");
        if (ud4 == 1)
        {
            unsigned long long const blocks = (total + 255) / 256;
            swap_flat_kernel<T><<<static_cast<unsigned>(blocks), 256, 0,
                ::nnhaul::cuda_stream()>>>(ud0, ud1, ud2, ud3, ud4, 0, total, st, dt);
        }
        else
        {
            unsigned long long const n_fibers = ud0 * ud1 * ud2 * ud3;
            unsigned long long const blocks =
                (n_fibers + kSwapWarps - 1) / kSwapWarps;
            dim3 const threads(32, kSwapWarps);
            swap_fiber_kernel<T><<<static_cast<unsigned>(blocks), threads,
                0, ::nnhaul::cuda_stream()>>>(ud0, ud1, ud2, ud3, ud4, 0, n_fibers, st,
                dt);
        }
    });
    check_launch("swap_two_axes");
}

void chunk_transpose(Buffer const &src, Buffer &dst, std::size_t m,
    std::size_t n, float alpha, ElemType type)
{
    require_kernel("chunk_transpose");
    if (m == 0 || n == 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    unsigned long long const um = m;
    unsigned long long const un = n;
    bool const fits = src.chunk_count()
            <= static_cast<std::size_t>(kMaxTableChunks)
        && dst.chunk_count() <= static_cast<std::size_t>(kMaxTableChunks);
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (!fits)
        {
            // Wide operands: destination tile grid split until each
            // piece's source/destination chunk spans fit the table.
            unsigned long long const it = (um + kTransTile - 1) / kTransTile;
            unsigned long long const jt = (un + kTransTile - 1) / kTransTile;
            transpose_emit_tiles<T>(um, un, alpha, es, 0, it, 0, jt, src,
                dst, "transpose");
            return;
        }
        ChunkTable const st = table_of(src, es, "transpose");
        ChunkTable const dt = table_of(dst, es, "transpose");
        unsigned long long const it = (um + kTransTile - 1) / kTransTile;
        unsigned long long const jt = (un + kTransTile - 1) / kTransTile;
        dim3 const blocks(static_cast<unsigned>(jt),
            static_cast<unsigned>(it));
        dim3 const threads(kTransTile, kTransTile);
        transpose_tile_kernel<T><<<blocks, threads, 0, ::nnhaul::cuda_stream()>>>(um,
            un, alpha, 0, 0, st, dt);
    });
    check_launch("transpose");
}

// ---------------------------------------------------------------------------
// conv2d host emitters: conservative source-position bounds of a
// contiguous destination flat range, then one launch or a split.
// ---------------------------------------------------------------------------

//! Digit ranges of a flat position range over dims/strides (the same
//! endpoint widening as swap_src_bounds: a dimension whose period is
//! fully covered takes its whole extent).
inline void chunk_digit_bounds(unsigned long long p0, unsigned long long p1,
    unsigned long long const *dims, unsigned long long const *strides,
    int n, unsigned long long *lo, unsigned long long *hi)
{
    unsigned long long const len = p1 - p0;
    for (int k = 0; k < n; ++k)
    {
        unsigned long long const dim = dims[k];
        unsigned long long const stride = strides[k];
        lo[k] = (p0 / stride) % dim;
        hi[k] = ((p1 - 1) / stride) % dim;
        if (len >= stride * dim)
        {
            lo[k] = 0;
            hi[k] = dim - 1;
        }
    }
}

inline bool chunk_span_fits(Buffer const &buf, std::size_t es,
    unsigned long long lo, unsigned long long hi, char const *who)
{
    return chunk_id_of(buf, es, hi, who) - chunk_id_of(buf, es, lo, who)
        < static_cast<unsigned long long>(kMaxTableChunks);
}

inline void chunk_launch_guard(unsigned long long p0, unsigned long long p1,
    char const *who)
{
    if (p1 - p0 != 1)
    {
        return;
    }
    std::fprintf(stderr,
        "[nnhaul] %s: one destination element spans more than %d source "
        "chunks; cannot build the by-value table\n",
        who, kMaxTableChunks);
    std::abort();
}

template <typename T>
void conv2d_emit_range(long long src1_m, long long src1_n,
    long long src1_channels, long long batch, long long src2_m,
    long long src2_n, long long dilation_m, long long dilation_n,
    long long dst_channels, long long offset_m, long long offset_n,
    float alpha, long long dst_m, long long dst_n, long long stride_m,
    long long stride_n, float beta, unsigned long long p0,
    unsigned long long p1, Buffer const &src1, Buffer const &src2,
    Buffer &dst, std::size_t es, char const *who)
{
    if (p0 >= p1)
    {
        return;
    }
    unsigned long long const dims[4] = {
        static_cast<unsigned long long>(dst_m),
        static_cast<unsigned long long>(dst_n),
        static_cast<unsigned long long>(dst_channels),
        static_cast<unsigned long long>(batch)};
    unsigned long long const strides[4] = {
        1,
        static_cast<unsigned long long>(dst_m),
        static_cast<unsigned long long>(dst_m) * dst_n,
        static_cast<unsigned long long>(dst_m) * dst_n * dst_channels};
    unsigned long long lo[4];
    unsigned long long hi[4];
    chunk_digit_bounds(p0, p1, dims, strides, 4, lo, hi);
    long long s1i_lo = std::max(0LL,
        static_cast<long long>(lo[0]) * stride_m - offset_m);
    long long s1i_hi = std::min(src1_m - 1,
        static_cast<long long>(hi[0]) * stride_m - offset_m
            + dilation_m * (src2_m - 1));
    long long s1j_lo = std::max(0LL,
        static_cast<long long>(lo[1]) * stride_n - offset_n);
    long long s1j_hi = std::min(src1_n - 1,
        static_cast<long long>(hi[1]) * stride_n - offset_n
            + dilation_n * (src2_n - 1));
    bool const reads = s1i_lo <= s1i_hi && s1j_lo <= s1j_hi;
    unsigned long long s1lo = 0;
    unsigned long long s1hi = 0;
    unsigned long long s2lo = 0;
    unsigned long long s2hi = 0;
    if (reads)
    {
        s1lo = static_cast<unsigned long long>(s1i_lo)
            + (lo[3] * static_cast<unsigned long long>(src1_channels)
                * src1_n + static_cast<unsigned long long>(s1j_lo))
                * src1_m;
        s1hi = static_cast<unsigned long long>(s1i_hi)
            + (hi[3] * static_cast<unsigned long long>(src1_channels)
                * src1_n + static_cast<unsigned long long>(s1j_hi))
                * src1_m
            + static_cast<unsigned long long>(src1_channels - 1)
                * src1_n * src1_m;
        unsigned long long const oc_step = static_cast<unsigned long long>(
            src2_n) * src2_m * src1_channels;
        s2lo = lo[2] * oc_step;
        s2hi = static_cast<unsigned long long>(src2_m - 1)
            + static_cast<unsigned long long>(src2_n - 1) * src2_m
            + hi[2] * oc_step
            + static_cast<unsigned long long>(src1_channels - 1)
                * src2_n * src2_m;
    }
    bool const fits =
        chunk_span_fits(dst, es, p0, p1 - 1, who)
        && chunk_span_fits(src1, es, s1lo, s1hi, who)
        && chunk_span_fits(src2, es, s2lo, s2hi, who);
    if (fits)
    {
        ChunkTable const s1t = make_table(src1, es,
            chunk_id_of(src1, es, s1lo, who),
            chunk_id_of(src1, es, s1hi, who) + 1, who);
        ChunkTable const s2t = make_table(src2, es,
            chunk_id_of(src2, es, s2lo, who),
            chunk_id_of(src2, es, s2hi, who) + 1, who);
        ChunkTable const dt = make_table(dst, es,
            chunk_id_of(dst, es, p0, who),
            chunk_id_of(dst, es, p1 - 1, who) + 1, who);
        unsigned long long const blocks = (p1 - p0 + 255) / 256;
        conv2d_inplace_chunk_kernel<T><<<static_cast<unsigned>(blocks),
            256, 0, ::nnhaul::cuda_stream()>>>(src1_m, src1_n, src1_channels, batch,
            src2_m, src2_n, dilation_m, dilation_n, dst_channels, offset_m,
            offset_n, alpha, dst_m, dst_n, stride_m, stride_n, beta, p0,
            p1, s1t, s2t, dt);
        return;
    }
    chunk_launch_guard(p0, p1, who);
    unsigned long long const mid = p0 + (p1 - p0) / 2;
    conv2d_emit_range<T>(src1_m, src1_n, src1_channels, batch, src2_m,
        src2_n, dilation_m, dilation_n, dst_channels, offset_m, offset_n,
        alpha, dst_m, dst_n, stride_m, stride_n, beta, p0, mid, src1,
        src2, dst, es, who);
    conv2d_emit_range<T>(src1_m, src1_n, src1_channels, batch, src2_m,
        src2_n, dilation_m, dilation_n, dst_channels, offset_m, offset_n,
        alpha, dst_m, dst_n, stride_m, stride_n, beta, mid, p1, src1,
        src2, dst, es, who);
}

template <typename T>
void conv2d_bwd_input_emit_range(long long src1_m, long long src1_n,
    long long stride_m, long long stride_n, long long src1_channels,
    long long batch, long long src2_m, long long src2_n,
    long long dilation_m, long long dilation_n, long long dst_channels,
    long long offset_m, long long offset_n, float alpha, long long dst_m,
    long long dst_n, float beta, unsigned long long p0,
    unsigned long long p1, Buffer const &src1, Buffer const &src2,
    Buffer &dst, std::size_t es, char const *who)
{
    if (p0 >= p1)
    {
        return;
    }
    unsigned long long const dims[4] = {
        static_cast<unsigned long long>(dst_m),
        static_cast<unsigned long long>(dst_n),
        static_cast<unsigned long long>(dst_channels),
        static_cast<unsigned long long>(batch)};
    unsigned long long const strides[4] = {
        1,
        static_cast<unsigned long long>(dst_m),
        static_cast<unsigned long long>(dst_m) * dst_n,
        static_cast<unsigned long long>(dst_m) * dst_n * dst_channels};
    unsigned long long lo[4];
    unsigned long long hi[4];
    chunk_digit_bounds(p0, p1, dims, strides, 4, lo, hi);
    long long s1i_lo = (static_cast<long long>(lo[0]) + offset_m
        - dilation_m * (src2_m - 1) + stride_m - 1) / stride_m;
    s1i_lo = std::max(s1i_lo, 0LL);
    long long s1i_hi = std::min(
        (static_cast<long long>(hi[0]) + offset_m + stride_m) / stride_m
            - 1, src1_m - 1);
    long long s1j_lo = (static_cast<long long>(lo[1]) + offset_n
        - dilation_n * (src2_n - 1) + stride_n - 1) / stride_n;
    s1j_lo = std::max(s1j_lo, 0LL);
    long long s1j_hi = std::min(
        (static_cast<long long>(hi[1]) + offset_n + stride_n) / stride_n
            - 1, src1_n - 1);
    bool const reads = s1i_lo <= s1i_hi && s1j_lo <= s1j_hi;
    unsigned long long s1lo = 0;
    unsigned long long s1hi = 0;
    unsigned long long s2lo = 0;
    unsigned long long s2hi = 0;
    if (reads)
    {
        s1lo = static_cast<unsigned long long>(s1i_lo)
            + (lo[3] * static_cast<unsigned long long>(src1_channels)
                * src1_n + static_cast<unsigned long long>(s1j_lo))
                * src1_m;
        s1hi = static_cast<unsigned long long>(s1i_hi)
            + (hi[3] * static_cast<unsigned long long>(src1_channels)
                * src1_n + static_cast<unsigned long long>(s1j_hi))
                * src1_m
            + static_cast<unsigned long long>(src1_channels - 1)
                * src1_n * src1_m;
        unsigned long long const oc_step = static_cast<unsigned long long>(
            src2_n) * src2_m * src1_channels;
        s2lo = lo[2] * oc_step;
        s2hi = static_cast<unsigned long long>(src2_m - 1)
            + static_cast<unsigned long long>(src2_n - 1) * src2_m
            + hi[2] * oc_step
            + static_cast<unsigned long long>(src1_channels - 1)
                * src2_n * src2_m;
    }
    bool const fits =
        chunk_span_fits(dst, es, p0, p1 - 1, who)
        && chunk_span_fits(src1, es, s1lo, s1hi, who)
        && chunk_span_fits(src2, es, s2lo, s2hi, who);
    if (fits)
    {
        ChunkTable const s1t = make_table(src1, es,
            chunk_id_of(src1, es, s1lo, who),
            chunk_id_of(src1, es, s1hi, who) + 1, who);
        ChunkTable const s2t = make_table(src2, es,
            chunk_id_of(src2, es, s2lo, who),
            chunk_id_of(src2, es, s2hi, who) + 1, who);
        ChunkTable const dt = make_table(dst, es,
            chunk_id_of(dst, es, p0, who),
            chunk_id_of(dst, es, p1 - 1, who) + 1, who);
        unsigned long long const blocks = (p1 - p0 + 255) / 256;
        conv2d_bwd_input_inplace_chunk_kernel<T><<<
            static_cast<unsigned>(blocks), 256, 0, ::nnhaul::cuda_stream()>>>(src1_m,
            src1_n, stride_m, stride_n, src1_channels, batch, src2_m,
            src2_n, dilation_m, dilation_n, dst_channels, offset_m,
            offset_n, alpha, dst_m, dst_n, beta, p0, p1, s1t, s2t, dt);
        return;
    }
    chunk_launch_guard(p0, p1, who);
    unsigned long long const mid = p0 + (p1 - p0) / 2;
    conv2d_bwd_input_emit_range<T>(src1_m, src1_n, stride_m, stride_n,
        src1_channels, batch, src2_m, src2_n, dilation_m, dilation_n,
        dst_channels, offset_m, offset_n, alpha, dst_m, dst_n, beta, p0,
        mid, src1, src2, dst, es, who);
    conv2d_bwd_input_emit_range<T>(src1_m, src1_n, stride_m, stride_n,
        src1_channels, batch, src2_m, src2_n, dilation_m, dilation_n,
        dst_channels, offset_m, offset_n, alpha, dst_m, dst_n, beta, mid,
        p1, src1, src2, dst, es, who);
}

template <typename T>
void conv2d_bwd_weight_emit_range(long long src1_m, long long src1_n,
    long long src1_channels, long long batch, long long src2_m,
    long long src2_n, long long stride_m, long long stride_n,
    long long src2_channels, long long offset_m, long long offset_n,
    float alpha, long long dst_m, long long dst_n, long long dilation_m,
    long long dilation_n, float beta, unsigned long long p0,
    unsigned long long p1, Buffer const &src1, Buffer const &src2,
    Buffer &dst, std::size_t es, char const *who)
{
    if (p0 >= p1)
    {
        return;
    }
    unsigned long long const dims[4] = {
        static_cast<unsigned long long>(dst_m),
        static_cast<unsigned long long>(dst_n),
        static_cast<unsigned long long>(src1_channels),
        static_cast<unsigned long long>(src2_channels)};
    unsigned long long const strides[4] = {
        1,
        static_cast<unsigned long long>(dst_m),
        static_cast<unsigned long long>(dst_m) * dst_n,
        static_cast<unsigned long long>(dst_m) * dst_n * src1_channels};
    unsigned long long lo[4];
    unsigned long long hi[4];
    chunk_digit_bounds(p0, p1, dims, strides, 4, lo, hi);
    long long s1i_lo = std::max(0LL,
        dilation_m * static_cast<long long>(lo[0]) - offset_m);
    long long s1i_hi = std::min(src1_m - 1,
        dilation_m * static_cast<long long>(hi[0]) - offset_m
            + stride_m * (src2_m - 1));
    long long s1j_lo = std::max(0LL,
        dilation_n * static_cast<long long>(lo[1]) - offset_n);
    long long s1j_hi = std::min(src1_n - 1,
        dilation_n * static_cast<long long>(hi[1]) - offset_n
            + stride_n * (src2_n - 1));
    bool const reads = s1i_lo <= s1i_hi && s1j_lo <= s1j_hi;
    unsigned long long s1lo = 0;
    unsigned long long s1hi = 0;
    unsigned long long s2lo = 0;
    unsigned long long s2hi = 0;
    if (reads)
    {
        s1lo = static_cast<unsigned long long>(s1i_lo)
            + (lo[2] * static_cast<unsigned long long>(src1_n)
                + static_cast<unsigned long long>(s1j_lo)) * src1_m;
        s1hi = static_cast<unsigned long long>(s1i_hi)
            + (hi[2] * static_cast<unsigned long long>(src1_n)
                + static_cast<unsigned long long>(s1j_hi)) * src1_m
            + static_cast<unsigned long long>(batch - 1)
                * src1_channels * src1_n * src1_m;
        s2lo = lo[3] * static_cast<unsigned long long>(src2_n) * src2_m;
        s2hi = static_cast<unsigned long long>(src2_m - 1)
            + (hi[3] * static_cast<unsigned long long>(src2_n)
                + src2_n - 1) * src2_m
            + static_cast<unsigned long long>(batch - 1)
                * src2_channels * src2_n * src2_m;
    }
    bool const fits =
        chunk_span_fits(dst, es, p0, p1 - 1, who)
        && chunk_span_fits(src1, es, s1lo, s1hi, who)
        && chunk_span_fits(src2, es, s2lo, s2hi, who);
    if (fits)
    {
        ChunkTable const s1t = make_table(src1, es,
            chunk_id_of(src1, es, s1lo, who),
            chunk_id_of(src1, es, s1hi, who) + 1, who);
        ChunkTable const s2t = make_table(src2, es,
            chunk_id_of(src2, es, s2lo, who),
            chunk_id_of(src2, es, s2hi, who) + 1, who);
        ChunkTable const dt = make_table(dst, es,
            chunk_id_of(dst, es, p0, who),
            chunk_id_of(dst, es, p1 - 1, who) + 1, who);
        unsigned long long const blocks = (p1 - p0 + 255) / 256;
        conv2d_bwd_weight_inplace_chunk_kernel<T><<<
            static_cast<unsigned>(blocks), 256, 0, ::nnhaul::cuda_stream()>>>(src1_m,
            src1_n, src1_channels, batch, src2_m, src2_n, stride_m,
            stride_n, src2_channels, offset_m, offset_n, alpha, dst_m,
            dst_n, dilation_m, dilation_n, beta, p0, p1, s1t, s2t, dt);
        return;
    }
    chunk_launch_guard(p0, p1, who);
    unsigned long long const mid = p0 + (p1 - p0) / 2;
    conv2d_bwd_weight_emit_range<T>(src1_m, src1_n, src1_channels, batch,
        src2_m, src2_n, stride_m, stride_n, src2_channels, offset_m,
        offset_n, alpha, dst_m, dst_n, dilation_m, dilation_n, beta, p0,
        mid, src1, src2, dst, es, who);
    conv2d_bwd_weight_emit_range<T>(src1_m, src1_n, src1_channels, batch,
        src2_m, src2_n, stride_m, stride_n, src2_channels, offset_m,
        offset_n, alpha, dst_m, dst_n, dilation_m, dilation_n, beta, mid,
        p1, src1, src2, dst, es, who);
}

template <typename T>
void randn_emit_range(unsigned long long p0, unsigned long long p1,
    RandnChunkGeom const &geom, Buffer &data, std::size_t es,
    char const *who)
{
    if (p0 >= p1)
    {
        return;
    }
    unsigned long long dims[kRandnMaxNdim];
    unsigned long long strides[kRandnMaxNdim];
    unsigned long long lo[kRandnMaxNdim];
    unsigned long long hi[kRandnMaxNdim];
    unsigned long long acc = 1;
    for (int d = 0; d < geom.ndim; ++d)
    {
        dims[d] = static_cast<unsigned long long>(geom.shape[d]);
        strides[d] = acc;
        acc *= dims[d];
    }
    chunk_digit_bounds(p0, p1, dims, strides,
        static_cast<int>(geom.ndim), lo, hi);
    long long o_min = 0;
    long long o_max = 0;
    for (int d = 0; d < geom.ndim; ++d)
    {
        long long const st = geom.stride[d];
        long long const a = static_cast<long long>(lo[d]) * st;
        long long const b = static_cast<long long>(hi[d]) * st;
        o_min += std::min(a, b);
        o_max += std::max(a, b);
    }
    if (o_min < 0 || o_max < 0)
    {
        std::fprintf(stderr,
            "[nnhaul] %s: negative output offsets are not supported\n",
            who);
        std::abort();
    }
    unsigned long long const ulo = static_cast<unsigned long long>(o_min);
    unsigned long long const uhi = static_cast<unsigned long long>(o_max);
    if (chunk_span_fits(data, es, ulo, uhi, who))
    {
        ChunkTable const dt = make_table(data, es,
            chunk_id_of(data, es, ulo, who),
            chunk_id_of(data, es, uhi, who) + 1, who);
        unsigned long long const blocks = (p1 - p0 + 255) / 256;
        randn_chunk_kernel<T><<<static_cast<unsigned>(blocks), 256, 0,
            ::nnhaul::cuda_stream()>>>(p0, p1, geom.seed, geom.mean, geom.stddev, geom,
            dt);
        return;
    }
    if (p1 - p0 == 1)
    {
        std::fprintf(stderr,
            "[nnhaul] %s: one output element spans more than %d chunks; "
            "cannot build the by-value table\n",
            who, kMaxTableChunks);
        std::abort();
    }
    unsigned long long const mid = p0 + (p1 - p0) / 2;
    randn_emit_range<T>(p0, mid, geom, data, es, who);
    randn_emit_range<T>(mid, p1, geom, data, es, who);
}

void chunk_conv2d_inplace(Buffer const &src1, Buffer const &src2,
    Buffer &dst, long long src1_m, long long src1_n,
    long long src1_channels, long long batch, long long src2_m,
    long long src2_n, long long dilation_m, long long dilation_n,
    long long dst_channels, long long offset_m, long long offset_n,
    float alpha, long long dst_m, long long dst_n, long long stride_m,
    long long stride_n, float beta, ElemType type)
{
    require_kernel("chunk_conv2d_inplace");
    if (src1_m <= 0 || src1_n <= 0 || src1_channels <= 0 || batch <= 0
        || src2_m <= 0 || src2_n <= 0 || dst_channels <= 0 || dst_m <= 0
        || dst_n <= 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    unsigned long long const total = static_cast<unsigned long long>(dst_m)
        * dst_n * dst_channels * batch;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv2d_emit_range<T>(src1_m, src1_n, src1_channels, batch, src2_m,
            src2_n, dilation_m, dilation_n, dst_channels, offset_m,
            offset_n, alpha, dst_m, dst_n, stride_m, stride_n, beta, 0,
            total, src1, src2, dst, es, "conv2d_inplace");
    });
    check_launch("conv2d_inplace");
}

void chunk_conv2d_bwd_input_inplace(Buffer const &src1,
    Buffer const &src2, Buffer &dst, long long src1_m, long long src1_n,
    long long stride_m, long long stride_n, long long src1_channels,
    long long batch, long long src2_m, long long src2_n,
    long long dilation_m, long long dilation_n, long long dst_channels,
    long long offset_m, long long offset_n, float alpha, long long dst_m,
    long long dst_n, float beta, ElemType type)
{
    require_kernel("chunk_conv2d_bwd_input_inplace");
    if (src1_m <= 0 || src1_n <= 0 || src1_channels <= 0 || batch <= 0
        || src2_m <= 0 || src2_n <= 0 || dst_channels <= 0 || dst_m <= 0
        || dst_n <= 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    unsigned long long const total = static_cast<unsigned long long>(dst_m)
        * dst_n * dst_channels * batch;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv2d_bwd_input_emit_range<T>(src1_m, src1_n, stride_m, stride_n,
            src1_channels, batch, src2_m, src2_n, dilation_m, dilation_n,
            dst_channels, offset_m, offset_n, alpha, dst_m, dst_n, beta,
            0, total, src1, src2, dst, es, "conv2d_bwd_input_inplace");
    });
    check_launch("conv2d_bwd_input_inplace");
}

void chunk_conv2d_bwd_weight_inplace(Buffer const &src1,
    Buffer const &src2, Buffer &dst, long long src1_m, long long src1_n,
    long long src1_channels, long long batch, long long src2_m,
    long long src2_n, long long stride_m, long long stride_n,
    long long src2_channels, long long offset_m, long long offset_n,
    float alpha, long long dst_m, long long dst_n, long long dilation_m,
    long long dilation_n, float beta, ElemType type)
{
    require_kernel("chunk_conv2d_bwd_weight_inplace");
    if (src1_m <= 0 || src1_n <= 0 || src1_channels <= 0 || batch <= 0
        || src2_m <= 0 || src2_n <= 0 || src2_channels <= 0 || dst_m <= 0
        || dst_n <= 0)
    {
        return;
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    unsigned long long const total = static_cast<unsigned long long>(dst_m)
        * dst_n * src1_channels * src2_channels;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv2d_bwd_weight_emit_range<T>(src1_m, src1_n, src1_channels,
            batch, src2_m, src2_n, stride_m, stride_n, src2_channels,
            offset_m, offset_n, alpha, dst_m, dst_n, dilation_m,
            dilation_n, beta, 0, total, src1, src2, dst, es,
            "conv2d_bwd_weight_inplace");
    });
    check_launch("conv2d_bwd_weight_inplace");
}

void chunk_randn(Buffer &data, std::size_t nelems,
    unsigned long long seed, float mean, float stddev, long long ndim,
    long long const *start, long long const *shape,
    long long const *stride, long long const *underlying_shape,
    ElemType type)
{
    require_kernel("chunk_randn");
    if (ndim <= 0 || ndim > kRandnMaxNdim || nelems == 0)
    {
        return;
    }
    RandnChunkGeom geom{};
    geom.ndim = ndim;
    geom.seed = seed;
    geom.mean = mean;
    geom.stddev = stddev;
    for (long long d = 0; d < ndim; ++d)
    {
        geom.start[d] = start[d];
        geom.shape[d] = shape[d];
        geom.stride[d] = stride[d];
        geom.underlying_shape[d] = underlying_shape[d];
    }
    std::size_t const es = type == ElemType::F64 ? 8
        : (type == ElemType::BF16 || type == ElemType::F16 ? 2 : 4);
    unsigned long long const total = nelems;
    dispatch_type(type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        randn_emit_range<T>(0, total, geom, data, es, "randn");
    });
    check_launch("randn");
}

} // namespace nntile::kernel::nnhaul
