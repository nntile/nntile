/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_inplace/cuda.cu
 * Add operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include <cstdio>
#include "nntile/kernel/add_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::add_inplace
{

template<typename T, int BLOCK, int LOOP>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T *src, Scalar beta_,
        T *dst)
//! Generic implementation of the add_inplace operation on CUDA
/*! @copydoc nntile::kernel::add_inplace::cuda
 * */
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    Y dst_block[LOOP];
    Y src_block[LOOP];
    Y alpha = Y{alpha_};
    Y beta = Y{beta_};
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < LOOP; ++j)
        {
            dst_block[j] = static_cast<Y>(dst[i+j*BLOCK_STEP]);
            src_block[j] = static_cast<Y>(src[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*src_block[j] + beta*dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
    else
    {
        int j_max = (nelems-i+BLOCK_STEP-1) / BLOCK_STEP;
        for(int j = 0; j < j_max; ++j)
        {
            dst_block[j] = static_cast<Y>(dst[i+j*BLOCK_STEP]);
            Y val1 = static_cast<Y>(src[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*val1 + beta*dst_block[j];
            dst[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha_, const T *src_,
        Scalar beta_, T *dst_)
    noexcept
//! Add two buffers inplace with optional scaling inplace on CUDA
/*! Performs the following operation:
 *      dst[i] = alpha*src[i] + beta*dst[i],
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and dst[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace instead, and in a case of beta being zero,
 * use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] stream: CUDA stream
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[in] beta_: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the add operation
 * */
{
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0, stream>>>(nelems, alpha_,
            src_, beta_, dst_);
}

//! Chunk descriptors for the one-launch chunked add_inplace (see cuda.hh).
template<typename T>
struct AddInplaceChunkTables
{
    const T* const* src; //!< device-ring view of the src pointer array
    Index nsrc;
    T* const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< log2(elems_src) on a power of two, else -1
    int shift_dst; //!< log2(elems_dst) on a power of two, else -1
    long nelems;
    Scalar alpha;
    Scalar beta;
};

//! Chunk index of a flat element offset (shift on a power-of-two elems).
static __device__ __forceinline__ Index addi_chunk_of(long f, Index elems,
    int shift)
{
    return Index(f >> shift);
}

//! Correctly-rounded FMA, to pin the association nvcc chose for cuda().
static __device__ __forceinline__ float addi_fma(float a, float b, float c)
{
    return __fmaf_rn(a, b, c);
}
static __device__ __forceinline__ double addi_fma(double a, double b,
    double c)
{
    return __fma_rn(a, b, c);
}

//! In-chunk element offset of flat offset f (mask when elems is pow2).
static __device__ __forceinline__ long addi_off(long f, Index elems, int shift,
    Index c)
{
    return f & ((1LL << shift) - 1);
}

//! Chunk-aware add_inplace: the SAME flat thread mapping and per-element
//! expression as cuda_kernel, with every operand address resolved through
//! its chunk table.
template<typename T, int BLOCK, int LOOP>
static __global__
void chunked_cuda_kernel(AddInplaceChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    const Y alpha = t.alpha;
    const Y beta = t.beta;
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    for(int j = 0; j < LOOP; ++j)
    {
        long f = long(i) + long(j)*BLOCK_STEP;
        if(f < t.nelems)
        {
            Index cs = addi_chunk_of(f, t.elems_src, t.shift_src);
            Y sv = Y{t.src[cs][addi_off(f, t.elems_src, t.shift_src, cs)]};
            Index cd = addi_chunk_of(f, t.elems_dst, t.shift_dst);
            T *dp = t.dst[cd] + addi_off(f, t.elems_dst, t.shift_dst, cd);
            // Same association as cuda_kernel: fl(alpha*src), then
            // fma(beta, dst, .).
            Y pr = alpha * sv;
            dp[0] = T{addi_fma(beta, Y{dp[0]}, pr)};
        }
    }
}

//! Chunked add_inplace with every operand inside its chunk 0 (each ring
//! holds one chunk, so every flat offset is a plain element offset of
//! ring[0]).  The two chunk-0 bases are resolved once per thread (two
//! warp-uniform loads hoisted out of the loop) and the body is
//! cuda_kernel's verbatim -- same flat thread mapping, same unrolled
//! loop, same per-element expression and therefore the same float
//! association nvcc chose for cuda() -- so the per-element instruction
//! stream matches the classic kernel's and the per-element ring loads
//! and shift/mask of the multi-chunk walk are gone.
template<typename T, int BLOCK, int LOOP>
static __global__
void chunked_cuda_kernel_one(Index nelems, Scalar alpha_, const T* const* src,
        Scalar beta_, T* const* dst)
{
    int i = threadIdx.x + blockIdx.x*BLOCK;
    using Y = typename T::repr_t;
    const T * const src_ = src[0];
    T * const dst_ = dst[0];
    Y dst_block[LOOP];
    Y src_block[LOOP];
    Y alpha = Y{alpha_};
    Y beta = Y{beta_};
    constexpr int BLOCK_STEP = BLOCK / LOOP;
    if((blockIdx.x+1)*BLOCK <= nelems)
    {
        for(int j = 0; j < LOOP; ++j)
        {
            dst_block[j] = static_cast<Y>(dst_[i+j*BLOCK_STEP]);
            src_block[j] = static_cast<Y>(src_[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*src_block[j] + beta*dst_block[j];
            dst_[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
    else
    {
        int j_max = (nelems-i+BLOCK_STEP-1) / BLOCK_STEP;
        for(int j = 0; j < j_max; ++j)
        {
            dst_block[j] = static_cast<Y>(dst_[i+j*BLOCK_STEP]);
            Y val1 = static_cast<Y>(src_[i+j*BLOCK_STEP]);
            dst_block[j] = alpha*val1 + beta*dst_block[j];
            dst_[i+j*BLOCK_STEP] = static_cast<T>(dst_block[j]);
        }
    }
}

//! Chunked add_inplace with every operand inside its chunk 0 and a
//! 4-multiple element count (4-byte element types only, dispatch guard
//! in chunked()): the flat space is walked in aligned float4 quads, one
//! quad per thread -- 16 bytes per load/store, a quarter of the memory
//! instructions of the scalar walk for exactly the same bytes.  Chunk
//! bases come from the engine's chunk allocator (cudaMalloc, at least
//! 256-byte aligned) and a single-chunk operand spans its whole chunk
//! from the base, so every quad access is 16-byte aligned.  The
//! per-element expression and its association are exactly cuda()'s:
//! fl(alpha*src) first, then a correctly-rounded fma(beta, dst, .).
template<typename T>
static __global__
void chunked_cuda_kernel_one_v4(Index nelems, Scalar alpha_,
        const T* const* src, Scalar beta_, T* const* dst)
{
    using Y = typename T::repr_t;
    using V = float4;
    long const q = long(threadIdx.x) + long(blockIdx.x)*long(blockDim.x);
    long const nquads = long(nelems) >> 2;
    if(q >= nquads)
    {
        return;
    }
    const Y alpha = Y{alpha_};
    const Y beta = Y{beta_};
    const V * const ps = reinterpret_cast<const V *>(src[0]) + q;
    V * const pd = reinterpret_cast<V *>(dst[0]) + q;
    V sv = ps[0];
    V dv = pd[0];
    V o;
    Y pr;
    pr = alpha * sv.x;
    o.x = addi_fma(beta, dv.x, pr);
    pr = alpha * sv.y;
    o.y = addi_fma(beta, dv.y, pr);
    pr = alpha * sv.z;
    o.z = addi_fma(beta, dv.z, pr);
    pr = alpha * sv.w;
    o.w = addi_fma(beta, dv.w, pr);
    pd[0] = o;
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src_chunks, Index nsrc, Scalar beta,
        T* const* dst_chunks, Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(nelems <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    if(nsrc == 1 and ndst == 1)
    {
        // Every operand inside its chunk 0 (each ring holds one chunk,
        // i.e. any contiguous buffer below the chunk class, e.g. every
        // GPT-2 add_inplace call).  A 4-multiple element count on a
        // 4-byte type takes the quad kernel; anything else takes the
        // classic body over the ring[0] bases, resolved once per thread.
        if constexpr(sizeof(T) == 4)
        {
            if((nelems & 3) == 0)
            {
                Index const nquads = nelems >> 2;
                dim3 threads(256);
                dim3 blocks(unsigned((nquads+255)/256));
                (chunked_cuda_kernel_one_v4<T>)<<<blocks, threads, 0,
                    stream>>>(nelems, alpha, src_chunks, beta, dst_chunks);
                return;
            }
        }
        dim3 threads(256);
        dim3 blocks((nelems+1023)/1024);
        (chunked_cuda_kernel_one<T, 1024, 4>)<<<blocks, threads, 0,
            stream>>>(nelems, alpha, src_chunks, beta, dst_chunks);
        return;
    }
    AddInplaceChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_add_inplace");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_add_inplace");
    t.nelems = long(nelems);
    t.alpha = alpha;
    t.beta = beta;
    dim3 threads(256);
    dim3 blocks((nelems+1023)/1024);
    (chunked_cuda_kernel<T, 1024, 4>)<<<blocks, threads, 0,
        stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src, Scalar beta, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src, Scalar beta, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src, Scalar beta, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *src, Scalar beta, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* src_chunks, Index nsrc, Scalar beta,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src_chunks, Index nsrc, Scalar beta,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src_chunks, Index nsrc, Scalar beta,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src_chunks, Index nsrc, Scalar beta,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;






} // namespace nntile::kernel::add_inplace
