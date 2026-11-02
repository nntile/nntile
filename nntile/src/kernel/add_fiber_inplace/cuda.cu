/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber_inplace/cuda.cu
 * Per-element addition of a tensor and a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_fiber_inplace/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_fiber_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src,
        Scalar beta_, T *dst)
//! Generic implementation of the add_fiber_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_fiber_inplace::cuda
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y beta{beta_};
    if(i2 < k and i1 < n and i0 < m)
    {
        for(Index b = 0; b < batch; ++b)
        {
            // Value to add to the output slice
            const Y src_val = alpha * Y{src[i2+b*k]};
            // Output fiber to be updated
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Read value from the output
            T &dst_val = dst_fiber[i0];
            // And update it
            dst_val = T{beta * Y{dst_val} + src_val};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src, Scalar beta, T *dst)
    noexcept
//! Add a broadcasted fiber into a tensor inplace with optional scaling on CUDA
/*! Performs the following operations:
 *      dst[i,l,j] = beta*dst[i,l,j] + alpha*src[l]
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[l,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If beta is zero and dst[i,l,j,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace, and in a case of beta being zero,
 * use nntile::kernel::scale_fiber instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale_fiber
 * @see nntile::kernel::clear
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[in] beta: Scaling factor for dst
 * @param[inout] dst: Input and output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src, beta, dst);
}

//! Chunk descriptors for the one-launch chunked add_fiber_inplace
//! (cuda.hh).  The per-operand chunk shifts are precomputed on the host
//! (they are the class shift for a multi-chunk operand, the
//! single-chunk sentinel 62 otherwise), and the ==1 facts are
//! precomputed bools, so the kernels never re-derive them and the
//! parameter block stays small.
template<typename T>
struct AddFiberInplaceChunkTables
{
    const T* const* src; //!< device-ring view of the src pointer array
    T* const* dst; //!< device-ring view of the dst pointer array
    bool one_s; //!< src is a single chunk (chunk 0 for every element)
    bool one_d; //!< dst is a single chunk
    int shift_src; //!< log2 of src full-chunk elems (62 if one chunk)
    int shift_dst; //!< log2 of dst full-chunk elems (62 if one chunk)
    Index m;
    Index n;
    Index k;
    Index batch;
    Scalar alpha;
    Scalar beta;
};

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float adfbi_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double adfbi_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! Chunked add_fiber_inplace with every operand inside its chunk 0 (each
//! ring holds one chunk, so every flat offset is a plain element offset
//! of ring[0]).  Geometry matches cuda(): thread .x walks the contiguous
//! i2 fiber (the src mode), blockIdx.y is i0 and blockIdx.z walks the
//! i1 rows with a grid stride so any m/n fits; each thread loops over
//! batch with all invariants hoisted, so the inner loop is pointer
//! bumps only and matches the cuda() inner loop instruction for
//! instruction.  The per-element expression and its association are
//! exactly cuda()'s: fl(beta*dst) first, then a correctly-rounded
//! fma(alpha, src, .).  dst is read even for beta == 0, as cuda_kernel
//! does; each element is read and written by its sole owning thread.
template<typename T>
struct AddFiberInplaceOneChunk
{
    const T* const* src;
    T* const* dst;
    Index m;
    Index n;
    Index k;
    Index batch;
    Scalar alpha;
    Scalar beta;
};

template<typename T>
static __global__
void chunked_cuda_kernel_one(AddFiberInplaceOneChunk<T> t)
{
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    Index const i2 = threadIdx.x + blockIdx.x*blockDim.x;
    // Hoisted per thread: the two chunk-0 bases (warp-uniform loads)
    // and the flat-offset stride of one batch step.
    const T * const ps_0 = t.src[0] + long(i2);
    T * const pd_0 = t.dst[0];
    long const fstep = long(t.m)*long(t.k)*long(t.n);
    if(i2 >= t.k)
    {
        return;
    }
    // i0 grid-strides over m and i1 over n, so any m/n fits the grid.
    for(Index i0 = blockIdx.y; i0 < t.m; i0 += gridDim.y)
    {
        for(Index i1 = blockIdx.z; i1 < t.n; i1 += gridDim.z)
        {
            // Flat element offset of (i0, i2, i1, b=0):
            // f = i0 + m*(i2 + k*(i1 + n*b)).
            long const f0 = long(i0) + long(t.m)*(long(i2)
                + long(t.k)*long(i1));
            const T *ps = ps_0;
            T *pd = pd_0 + f0;
            for(Index b = 0; b < t.batch; ++b)
            {
                // Same association as cuda_kernel: fl(beta*dst), then
                // fma(alpha, src, .).
                Y const pr = beta * static_cast<Y>(pd[0]);
                pd[0] = static_cast<T>(adfbi_fma(alpha,
                    static_cast<Y>(ps[0]), pr));
                ps += long(t.k);
                pd += fstep;
            }
        }
    }
}

//! Chunked add_fiber_inplace with m == 1, k a multiple of 4 and every
//! operand inside its chunk 0 (each ring holds one chunk).  m == 1 makes
//! the i2 fiber the contiguous mode, so a thread can own one aligned quad
//! of 4 consecutive elements and move 16 bytes per load/store instead
//! of 4 -- a quarter of the memory instructions of the scalar walk for
//! exactly the same bytes and exactly the same per-element expression
//! (each lane keeps cuda()'s association: fl(beta*dst) first, then a
//! correctly-rounded fma(alpha, src, .)).  The quad is read and written
//! in place by that one thread, its sole owner.  Only reachable for
//! 4-byte element types (dispatch guard in chunked()).
template<typename T>
static __global__
void chunked_cuda_kernel_one_v4(AddFiberInplaceOneChunk<T> t)
{
    using Y = typename T::repr_t;
    using V = float4;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    // Quad index inside the i2 fiber (k/4 quads), guarded because the
    // last block along x may stick out.
    Index const q = threadIdx.x + blockIdx.x*blockDim.x;
    Index const kq = t.k >> 2;
    if(q >= kq)
    {
        return;
    }
    // Quads per batch step of the flat space (k*n elements).
    long const rowq = long(kq)*long(t.n);
    // Chunk-0 bases, hoisted once per thread (warp-uniform loads).
    const V * const ps_0 = reinterpret_cast<const V *>(t.src[0]) + q;
    V * const pd_0 = reinterpret_cast<V *>(t.dst[0]) + q;
    // batch is split across gridDim.y and the i1 rows across the z
    // grid blocks (grid stride), so any batch/n fits and the launch
    // covers the whole element space with full warps.
    for(Index b = blockIdx.y; b < t.batch; b += gridDim.y)
    {
        long const boff = long(b)*rowq;
        for(Index i1 = Index(blockIdx.z)*blockDim.y + threadIdx.y;
                i1 < t.n; i1 += Index(gridDim.z)*blockDim.y)
        {
            long const off = long(i1)*long(kq) + boff;
            V const s1 = ps_0[long(b)*long(kq)];
            V const d = pd_0[off];
            V o;
            o.x = adfbi_fma(alpha, s1.x, beta*d.x);
            o.y = adfbi_fma(alpha, s1.y, beta*d.y);
            o.z = adfbi_fma(alpha, s1.z, beta*d.z);
            o.w = adfbi_fma(alpha, s1.w, beta*d.w);
            pd_0[off] = o;
        }
    }
}

//! Chunked add_fiber_inplace with at least one operand spanning several
//! chunks.  Same geometry as chunked_cuda_kernel_one, but each element's
//! chunk is resolved from its flat offset by a pure shift/mask on the
//! chunk class (never a runtime division) and the chunk base is read
//! from the ring; per-thread invariants -- the hoisted chunk-0 bases,
//! the batch-stride, the src step and the shift/mask constants -- stay
//! out of the inner loop.  dst is read and written in place, once per
//! element, by its sole owning thread.  Same per-element expression as
//! cuda().
template<typename T>
static __global__
void chunked_cuda_kernel_many(AddFiberInplaceChunkTables<T> t)
{
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    Index const i2 = threadIdx.x + blockIdx.x*blockDim.x;
    const T * const ps_0 = t.src[0];
    T * const pd_0 = t.dst[0];
    long const fstep = long(t.m)*long(t.k)*long(t.n);
    long const ms = (1LL << t.shift_src) - 1;
    long const md = (1LL << t.shift_dst) - 1;
    if(i2 >= t.k)
    {
        return;
    }
    // i0 grid-strides over m and i1 over n, so any m/n fits the grid.
    for(Index i0 = blockIdx.y; i0 < t.m; i0 += gridDim.y)
    {
        for(Index i1 = blockIdx.z; i1 < t.n; i1 += gridDim.z)
        {
            // Flat element offsets of (i0, i2, i1, b=0): dst's
            // f = i0 + m*(i2 + k*(i1 + n*b)); src's f1 = i2 + k*b.
            long f = long(i0) + long(t.m)*(long(i2)
                + long(t.k)*long(i1));
            long f1 = long(i2);
            for(Index b = 0; b < t.batch; ++b)
            {
                // A single-chunk operand keeps its hoisted chunk-0
                // base; a multi-chunk one reads the chunk base from the
                // ring.
                const T * const ps = t.one_s ? ps_0
                    : t.src[Index(f1 >> t.shift_src)];
                Y const sv = static_cast<Y>(ps[f1 & ms]);
                Index const cd = Index(f >> t.shift_dst);
                T * const dp = t.one_d ? pd_0 : t.dst[cd];
                // Same association as cuda_kernel: fl(beta*dst), then
                // fma(alpha, src, .).
                Y const pr = beta * static_cast<Y>(dp[f & md]);
                dp[f & md] = static_cast<T>(adfbi_fma(alpha, sv, pr));
                f += fstep;
                f1 += long(t.k);
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    // cuda()'s geometry: thread .x is the contiguous i2 fiber, blockIdx.y
    // is i0 and blockIdx.z walks the i1 rows (grid-stride, so any m/n
    // fits).  Each thread then loops over batch with every chunk and
    // stride invariant hoisted, which is what closes the gap to cuda()
    // on the GPT-2 shapes -- the generic path paid two ring-table
    // loads plus the shift/mask chunk resolution per element.
    int const tx = int(k < 1024 ? k : 1024);
    dim3 threads(tx, 1, 1);
    dim3 blocks(unsigned((k + tx - 1)/tx),
            unsigned(m < 65535 ? m : 65535),
            unsigned(n < 65535 ? n : 65535));
    if constexpr(sizeof(T) == 4)
    {
        // m == 1 with a 4-multiple k and every operand inside its
        // chunk 0 (any such tensor below the chunk class): the quad
        // kernel moves 16 bytes per access with the same per-element
        // expression.  Block: quads along x, a few i1 rows along y;
        // batch is split across gridDim.y, i1 rows across gridDim.z.
        if(m == 1 and (k & 3) == 0 and nsrc == 1 and ndst == 1)
        {
            Index const kq = k >> 2;
            int const qx = int(kq < 128 ? kq : 128);
            int const ry = int(std::min<Index>(std::max<Index>(128/qx, 1),
                n));
            dim3 qthreads(qx, ry, 1);
            // gridDim.z is clamped at the hardware limit 65535 (the
            // i1 loop grid-strides, so any n is still covered).
            Index const nz = (n + ry - 1)/ry;
            dim3 qblocks(unsigned((kq + qx - 1)/qx),
                    unsigned(batch < 65535 ? batch : 65535),
                    unsigned(nz < 65535 ? nz : 65535));
            AddFiberInplaceOneChunk<T> t;
            t.src = src_chunks;
            t.dst = dst_chunks;
            t.m = m;
            t.n = n;
            t.k = k;
            t.batch = batch;
            t.alpha = alpha;
            t.beta = beta;
            (chunked_cuda_kernel_one_v4<T>)<<<qblocks, qthreads, 0,
                stream>>>(t);
            return;
        }
    }
    if(nsrc == 1 and ndst == 1)
    {
        // Every operand inside its chunk 0 (any tensor below the chunk
        // class, e.g. every GPT-2 add_fiber_inplace call): the minimal
        // pointer-bump kernel, cuda()'s inner loop plus two ring
        // dereferences per thread.
        AddFiberInplaceOneChunk<T> t;
        t.src = src_chunks;
        t.dst = dst_chunks;
        t.m = m;
        t.n = n;
        t.k = k;
        t.batch = batch;
        t.alpha = alpha;
        t.beta = beta;
        (chunked_cuda_kernel_one<T>)<<<blocks, threads, 0, stream>>>(t);
        return;
    }
    AddFiberInplaceChunkTables<T> t;
    t.src = src_chunks;
    t.dst = dst_chunks;
    t.one_s = (nsrc == 1);
    t.one_d = (ndst == 1);
    t.shift_src = chunk_shift_of<T>(nsrc, elems_src, "nntile_add_fiber_inplace");
    t.shift_dst = chunk_shift_of<T>(ndst, elems_dst, "nntile_add_fiber_inplace");
    t.m = m;
    t.n = n;
    t.k = k;
    t.batch = batch;
    t.alpha = alpha;
    t.beta = beta;
    (chunked_cuda_kernel_many<T>)<<<blocks, threads, 0, stream>>>(t);
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






} // namespace nntile::kernel::add_fiber_inplace
