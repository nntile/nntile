/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_fiber/cuda.cu
 * Per-element addition of a tensor and a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_fiber
{

template<typename T>
static __global__
void cuda_kernel(
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha_,
    const T *src1,
    Scalar beta_,
    const T *src2,
    T *dst
)
//! Generic implementation of the add_fiber operation on CUDA
/*! @copydoc nntile::kernel::add_fiber::cuda
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
            const Y src1_val = alpha * static_cast<Y>(src1[i2+b*k]);
	        const T *src2_fiber = src2 + ((i1+b*n)*k+i2)*m;
            // Output fiber to be updated
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Read value from the output
            const Y src2_val = static_cast<Y>(src2_fiber[i0]);
            // And update output
            dst_fiber[i0] = static_cast<T>(beta * src2_val + src1_val);
        }
    }
}

template<typename T>
void cuda(
    cudaStream_t stream,
    Index m,
    Index n,
    Index k,
    Index batch,
    Scalar alpha,
    const T *src1,
    Scalar beta,
    const T *src2,
    T *dst
) noexcept
//! Add a tensor and a broadcasted fiber with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i,l,j,b] = beta*src2[i,l,j,b] + alpha*src1[l,b]
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[l,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If beta is zero and src2[i,l,j,b] is NaN, then dst[i,l,j,b] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale, and in a case of beta being zero,
 * use nntile::kernel::scale_fiber instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::scale_fiber
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of dst tensor.
 * @param[in] n: Size of the last mode of dst tensor.
 * @param[in] k: Size of the middle mode of dst and src2 tensors, and the first
 * mode of src1.
 * @param[in] batch: Size of the batch dimension.
 * @param[in] alpha: Scalar factor for src1.
 * @param[in] src1: Input contiguous vector with k*batch elements.
 * @param[in] beta: Scaling factor for src2.
 * @param[in] src2: Input contiguous tensor with m*k*n*batch elements.
 * @param[inout] dst: Output contiguous m-by-k-by-n-by-batch tensor.
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha,
            src1, beta, src2, dst);
}

//! Chunk descriptors for the one-launch chunked add_fiber (see cuda.hh).
//! The per-operand chunk shifts are precomputed on the host (they are
//! the class shift for a multi-chunk operand, the single-chunk sentinel
//! 62 otherwise), and the ==1 / dst-reuse facts are precomputed bools,
//! so the kernels never re-derive them and the parameter block stays
//! small.
template<typename T>
struct AddFiberChunkTables
{
    const T* const* src1; //!< device-ring view of the src1 pointer array
    const T* const* src2; //!< device-ring view of the src2 pointer array
    T* const* dst; //!< device-ring view of the dst pointer array
    bool one1; //!< src1 is a single chunk (chunk 0 for every element)
    bool one2; //!< src2 is a single chunk
    bool one_d; //!< dst is a single chunk
    bool dst_reuse; //!< src2 and dst share a chunking, so chunk idx too
    int shift_src1; //!< log2 of src2 full-chunk elems (62 if one chunk)
    int shift_src2; //!< log2 of src2 full-chunk elems (62 if one chunk)
    int shift_dst; //!< log2 of dst full-chunk elems (62 if one chunk)
    Index m;
    Index n;
    Index k;
    Index batch;
    Scalar alpha;
    Scalar beta;
};

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float adfb_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double adfb_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! Chunked add_fiber with every operand inside its chunk 0 (each ring
//! holds one chunk, so every flat offset is a plain element offset of
//! ring[0]).  Geometry matches cuda(): thread .x walks the contiguous
//! i2 fiber (the src1 mode), blockIdx.y is i0 and blockIdx.z walks the
//! i1 rows with a grid stride so any n fits; each thread loops over
//! batch with all invariants hoisted, so the inner loop is pointer
//! bumps only and matches the cuda() inner loop instruction for
//! instruction.  The per-element expression and its association are
//! exactly cuda()'s: fl(beta*src2) first, then a correctly-rounded
//! fma(alpha, src1, .).
template<typename T>
struct AddFiberOneChunk
{
    const T* const* src1;
    const T* const* src2;
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
void chunked_cuda_kernel_one(AddFiberOneChunk<T> t)
{
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    Index const i2 = threadIdx.x + blockIdx.x*blockDim.x;
    // Hoisted per thread: the three chunk-0 bases (warp-uniform loads)
    // and the flat-offset stride of one batch step.
    const T * const ps1_0 = t.src1[0] + long(i2);
    const T * const ps2_0 = t.src2[0];
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
            const T *ps2 = ps2_0 + f0;
            T *pd = pd_0 + f0;
            const T *ps1 = ps1_0;
            for(Index b = 0; b < t.batch; ++b)
            {
                // Same association as cuda_kernel: fl(beta*src2), then
                // fma(alpha, src1, .).
                Y const pr = beta * static_cast<Y>(ps2[0]);
                Y const out = adfb_fma(alpha, static_cast<Y>(ps1[0]),
                    pr);
                pd[0] = static_cast<T>(out);
                ps1 += long(t.k);
                ps2 += fstep;
                pd += fstep;
            }
        }
    }
}

//! Chunked add_fiber with m == 1, k a multiple of 4 and every operand
//! inside its chunk 0 (each ring holds one chunk).  m == 1 makes the
//! i2 fiber the contiguous mode, so a thread can own one aligned quad
//! of 4 consecutive elements and move 16 bytes per load/store instead
//! of 4 -- a quarter of the memory instructions of the scalar walk for
//! exactly the same bytes and exactly the same per-element expression
//! (each lane keeps cuda()'s association: fl(beta*src2) first, then a
//! correctly-rounded fma(alpha, src1, .)).  Only reachable for
//! 4-byte element types (dispatch guard in chunked()).
template<typename T>
static __global__
void chunked_cuda_kernel_one_v4(AddFiberOneChunk<T> t)
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
    const V * const ps1_0 = reinterpret_cast<const V *>(t.src1[0]) + q;
    const V * const ps2_0 = reinterpret_cast<const V *>(t.src2[0]) + q;
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
            V const s1 = ps1_0[long(b)*long(kq)];
            V const s2 = ps2_0[off];
            V o;
            o.x = adfb_fma(alpha, s1.x, beta*s2.x);
            o.y = adfb_fma(alpha, s1.y, beta*s2.y);
            o.z = adfb_fma(alpha, s1.z, beta*s2.z);
            o.w = adfb_fma(alpha, s1.w, beta*s2.w);
            pd_0[off] = o;
        }
    }
}

//! Chunked add_fiber with at least one operand spanning several chunks.
//! Same geometry as chunked_cuda_kernel_one, but each element's chunk
//! is resolved from its flat offset by a pure shift/mask on the chunk
//! class (never a runtime division) and the chunk base is read from the
//! ring; per-thread invariants -- the hoisted chunk-0 bases, the
//! batch-stride, the src1 step and the shift/mask constants -- stay out
//! of the inner loop.  Same per-element expression as cuda().
template<typename T>
static __global__
void chunked_cuda_kernel_many(AddFiberChunkTables<T> t)
{
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    Index const i2 = threadIdx.x + blockIdx.x*blockDim.x;
    const T * const p1_0 = t.src1[0];
    const T * const p2_0 = t.src2[0];
    T * const pd_0 = t.dst[0];
    long const fstep = long(t.m)*long(t.k)*long(t.n);
    long const m1 = (1LL << t.shift_src1) - 1;
    long const m2 = (1LL << t.shift_src2) - 1;
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
            // Flat element offset of (i0, i2, i1, b=0), and src1's
            // flat offset i2 + k*b.
            long f = long(i0) + long(t.m)*(long(i2)
                + long(t.k)*long(i1));
            long f1 = long(i2);
            for(Index b = 0; b < t.batch; ++b)
            {
                // A single-chunk operand keeps its hoisted chunk-0
                // base; a multi-chunk one reads the chunk base from the
                // ring.
                const T * const ps1 = t.one1 ? p1_0
                    : t.src1[Index(f1 >> t.shift_src1)];
                Y const s1 = static_cast<Y>(ps1[f1 & m1]);
                Index const c2 = Index(f >> t.shift_src2);
                const T * const ps2 = t.one2 ? p2_0 : t.src2[c2];
                Y const pr = beta * static_cast<Y>(ps2[f & m2]);
                Y const out = adfb_fma(alpha, s1, pr);
                // src2 and dst share a chunking in every driver call
                // site; when their chunk_elems match, dst's chunk index
                // and in-chunk offset are src2's.
                T *based;
                if(t.dst_reuse)
                {
                    based = t.one_d ? pd_0 : t.dst[c2];
                }
                else
                {
                    Index const cd = Index(f >> t.shift_dst);
                    based = t.one_d ? pd_0 : t.dst[cd];
                }
                based[f & md] = static_cast<T>(out);
                f += fstep;
                f1 += long(t.k);
            }
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T* const* src1_chunks, Index nsrc1, Scalar beta,
        const T* const* src2_chunks, Index nsrc2, T* const* dst_chunks,
        Index ndst, Index elems_src1, Index elems_src2, Index elems_dst)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0 || nsrc1 <= 0 || nsrc2 <= 0
        || ndst <= 0)
    {
        return;
    }
    // cuda()'s geometry: thread .x is the contiguous i2 fiber, blockIdx.y
    // is i0 and blockIdx.z walks the i1 rows (grid-stride, so any n
    // fits).  Each thread then loops over batch with every chunk and
    // stride invariant hoisted, which is what closes the gap to cuda()
    // on the GPT-2 shapes -- the earlier flat walk over f paid two
    // runtime 64-bit divisions (f/k, then /n) plus three ring-table
    // loads per element.
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
        if(m == 1 and (k & 3) == 0 and nsrc1 == 1 and nsrc2 == 1
            and ndst == 1)
        {
            Index const kq = k >> 2;
            int const qx = int(kq < 128 ? kq : 128);
            int const ry = int(std::min<Index>(std::max<Index>(128/qx, 1),
                n));
            dim3 qthreads(qx, ry, 1);
            dim3 qblocks(unsigned((kq + qx - 1)/qx),
                    unsigned(batch < 65535 ? batch : 65535),
                    unsigned((n + ry - 1)/ry));
            AddFiberOneChunk<T> t;
            t.src1 = src1_chunks;
            t.src2 = src2_chunks;
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
    if(nsrc1 == 1 and nsrc2 == 1 and ndst == 1)
    {
        // Every operand inside its chunk 0 (any tensor below the chunk
        // class, e.g. every GPT-2 add_fiber call): the minimal
        // pointer-bump kernel, cuda()'s inner loop plus one ring
        // dereference per thread.
        AddFiberOneChunk<T> t;
        t.src1 = src1_chunks;
        t.src2 = src2_chunks;
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
    AddFiberChunkTables<T> t;
    t.src1 = src1_chunks;
    t.src2 = src2_chunks;
    t.dst = dst_chunks;
    t.one1 = (nsrc1 == 1);
    t.one2 = (nsrc2 == 1);
    t.one_d = (ndst == 1);
    t.shift_src1 = chunk_shift_of<T>(nsrc1, elems_src1, "nntile_add_fiber");
    t.shift_src2 = chunk_shift_of<T>(nsrc2, elems_src2, "nntile_add_fiber");
    t.shift_dst = chunk_shift_of<T>(ndst, elems_dst, "nntile_add_fiber");
    t.dst_reuse = (elems_src2 == elems_dst
        && t.shift_src2 == t.shift_dst);
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
        Scalar alpha, const fp32_t *src1, Scalar beta,
	const fp32_t *src2, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src1, Scalar beta,
	const fp64_t *src2, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src1, Scalar beta,
	const bf16_t *src2, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src1, Scalar beta,
	const fp16_t *src2, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp32_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const fp32_t* const* src2_chunks,
        Index nsrc2, fp32_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp64_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const fp64_t* const* src2_chunks,
        Index nsrc2, fp64_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const bf16_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const bf16_t* const* src2_chunks,
        Index nsrc2, bf16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp16_t* const* src1_chunks,
        Index nsrc1, Scalar beta, const fp16_t* const* src2_chunks,
        Index nsrc2, fp16_t* const* dst_chunks, Index ndst, Index elems_src1,
        Index elems_src2, Index elems_dst) noexcept;






} // namespace nntile::kernel::add_fiber
