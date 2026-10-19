/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sum_fiber/cuda.cu
 * Sums over slices into a fiber of a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/sum_fiber/cuda.hh"
#include <cmath>
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::sum_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha, const T *src,
        Scalar beta, T *dst)
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k,b] = beta*dst[k,b] + alpha*sum(src[:,k,:,b])
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] sum: Output contiguous vector with k elements, that accumulate
 *      sums over slices along the first and the last axes.
 * */
{
    Index i2_batched = threadIdx.x + blockIdx.x*blockDim.x;
    Index i2 = i2_batched % k;
    Index b = i2_batched / k;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};
    // Init sum
    Y sum = zero;
    if(b < batch)
    {
        // Cycle over the third axis of input buffer
        for(Index i1 = i1_start; i1 < n; i1 += i1_step)
        {
            // Get sum of a corresponding slice
            const T *src_slice = src + ((i1+b*n)*k+i2)*m;
            // Cycle over the first axis of input buffer
            for(Index i0 = i0_start; i0 < m; i0 += i0_step)
            {
                // Read value from source
                T val = src_slice[i0];
                // Update sum
                sum += Y{val};
            }
        }
        __shared__ Y block_sum[1];
        if(i1_start == 0 and i0_start == 0)
        {
            block_sum[threadIdx.x] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x], sum);
        __syncthreads();
        // Update output value
        if(i1_start == 0 and i0_start == 0)
        {
            // Save result
            sum = block_sum[threadIdx.x];
            if(beta == zero)
            {
                sum *= alpha;
            }
            else
            {
                sum = beta * Y{dst[i2_batched]} + alpha * sum;
            }
            dst[i2_batched] = T{sum};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src, Scalar beta, T *dst)
    noexcept
//! Sums over slices along the first and last axes into a fiber of a tensor
/*! For a provided m-by-k-by-n input array computes sums over slices
 * along the first axis with m elements and the last axis with n elements,
 * resulting in output fiber of shape (k).
 * Mnemonically, the following operations are performed:
 *      dst[k,b] = beta*dst[k,b] + alpha*sum(src[:,k,:,b])
 *
 * @param[in] m: Size of the first mode of src array
 * @param[in] n: Size of the last mode of src array
 * @param[in] k: Size of the middle mode of src array and the only mode of
 *      dst array
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scaling factor for src
 * @param[in] src: Input contiguous m-by-k-by-n array
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Output contiguous vector with k elements, that accumulate
 *      sums over slices along the first and the last axes.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(1, std::min(int(m), 32), std::min(int(n), 32));
    dim3 blocks((k*batch+threads.x-1)/threads.x, 1, 1);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src, beta, dst);
}

//! Chunk descriptors for the one-launch chunked sum_fiber (see cuda.hh).
template<typename T>
struct SumFiberChunkTables
{
    const T* const* src; //!< device-ring view of the src chunk-pointer array
    T* const* dst; //!< device-ring view of the dst chunk-pointer array
    Index nsrc;
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    Index m;
    Index n;
    Index k;
    Index batch;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ Index sf_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Chunk-aware sum_fiber: the SAME thread/block geometry and the SAME
//! per-thread accumulation order (i1 outer, i0 inner) as cuda_kernel above,
//! with each element's src (and the writer's dst) resolved through its
//! chunk table. Because the value sequence each thread accumulates is
//! identical, the shared-memory atomicAdd reduction produces cuda()'s
//! result up to the same intra-block atomic ordering nondeterminism cuda()
//! already has.
template<typename T>
static __global__
void chunked_cuda_kernel(SumFiberChunkTables<T> t)
{
    Index i2_batched = threadIdx.x + blockIdx.x*blockDim.x;
    Index i2 = i2_batched % t.k;
    Index b = i2_batched / t.k;
    Index i0_start = threadIdx.y, i0_step = blockDim.y;
    Index i1_start = threadIdx.z, i1_step = blockDim.z;
    using Y = typename T::repr_t;
    constexpr Y zero{0.0};
    // Init sum
    Y sum = zero;
    if(b < t.batch)
    {
        // Run-structured in-chunk walk (piece 4): the double loop reads,
        // per thread, i1-major with the i0 fiber inside each i1 step.
        // Walk the outer axis through a virtual contiguous base
        // q = chunk_base - chunk*elems (the element at flat f is q[f]),
        // rebasing (one divmod) only between runs; the span-start
        // divmod runs once per thread. A run is cut so that every i1
        // step it covers has its whole i0 span [base, base+m) inside
        // one chunk; an i1 step whose i0 span itself straddles a chunk
        // boundary falls back to the per-element resolution (same
        // arithmetic as before). Same elements in the same order, so
        // the per-thread accumulation is unchanged bitwise.
        long const stride = long(i1_step) * long(t.k) * long(t.m);
        long f1 = (long(i1_start) + long(b)*long(t.n)) * long(t.k)
            * long(t.m) + long(i2) * long(t.m);
        long const i1_end = (long(t.n) + long(b)*long(t.n)) * long(t.k)
            * long(t.m) + long(i2) * long(t.m);
        long c1 = f1 >> t.shift_src;
        long off1 = f1 - c1 * t.elems_src;
        T const *q1 = t.src[c1] - c1 * t.elems_src;
        while(f1 < i1_end)
        {
            // Outer steps whose base AND full i0 span stay in-chunk.
            long room = (t.elems_src - off1 - long(t.m)) / stride + 1;
            if(room < 1)
            {
                room = 1;
            }
            long run_end = f1 + room * stride;
            if(run_end > i1_end)
            {
                run_end = i1_end;
            }
            for(; f1 < run_end; f1 += stride)
            {
                if(off1 + long(t.m) <= t.elems_src)
                {
                    // Cycle over the first axis of input buffer
                    for(Index i0 = i0_start; i0 < t.m; i0 += i0_step)
                    {
                        sum += Y{q1[f1 + long(i0)]};
                    }
                }
                else
                {
                    // Straddling i0 span: per-element resolution.
                    for(Index i0 = i0_start; i0 < t.m; i0 += i0_step)
                    {
                        long const f = f1 + long(i0);
                        Index const c = sf_chunk_of(f, t.elems_src,
                            t.shift_src);
                        sum += Y{t.src[c][f - long(c) * long(t.elems_src)]};
                    }
                }
                off1 += stride;
            }
            if(f1 < i1_end)
            {
                c1 = f1 >> t.shift_src;
                off1 = f1 - c1 * t.elems_src;
                q1 = t.src[c1] - c1 * t.elems_src;
            }
        }
        __shared__ Y block_sum[1];
        if(i1_start == 0 and i0_start == 0)
        {
            block_sum[threadIdx.x] = zero;
        }
        __syncthreads();
        atomicAdd(&block_sum[threadIdx.x], sum);
        __syncthreads();
        // Update output value
        if(i1_start == 0 and i0_start == 0)
        {
            // Save result
            sum = block_sum[threadIdx.x];
            long const g = long(i2_batched);
            Index const cd = sf_chunk_of(g, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + (g - long(cd) * long(t.elems_dst));
            if(t.beta == zero)
            {
                sum *= t.alpha;
            }
            else
            {
                sum = t.beta * Y{dp[0]} + t.alpha * sum;
            }
            dp[0] = T{sum};
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 or n <= 0 or k <= 0 or batch <= 0 or nsrc <= 0 or ndst <= 0)
    {
        return;
    }
    SumFiberChunkTables<T> t;
    t.src = src_chunks;
    t.dst = dst_chunks;
    t.nsrc = nsrc;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_sum_fiber");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_sum_fiber");
    t.m = m;
    t.n = n;
    t.k = k;
    t.batch = batch;
    t.alpha = alpha;
    t.beta = beta;
    // Same thread geometry as the whole-buffer entry.
    dim3 threads(1, std::min(int(m), 32), std::min(int(n), 32));
    dim3 blocks((k*batch+threads.x-1)/threads.x, 1, 1);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp32_t* const* src_chunks, Index nsrc,
        Scalar beta, fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp64_t* const* src_chunks, Index nsrc,
        Scalar beta, fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const bf16_t* const* src_chunks, Index nsrc,
        Scalar beta, bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp16_t* const* src_chunks, Index nsrc,
        Scalar beta, fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

} // namespace nntile::kernel::sum_fiber
