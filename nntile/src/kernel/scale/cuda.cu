/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/scale/cuda.cu
 * Scale operation on buffers on CUDA
 *
 * @version 1.3.0
 * */

#include "nntile/kernel/scale/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::scale
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, const T* src, T* dst)
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      dst[i] = alpha * src[i]
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha_: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[out] dst_: Destination of the scale operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i < nelems)
    {
        dst[i] = alpha * Y{src[i]};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src_,
        T *dst_)
    noexcept
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      dst[i] = alpha * src[i]
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src_: Source tensor
 * @param[out] dst_: Destination of the scale operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src_,
            dst_);
}

//! Chunk descriptors for the one-launch chunked scale (see cuda.hh).
template<typename T>
struct ChunkTables
{
    const T *const* src; //!< device-ring view of the src pointer array
    Index nsrc;
    T *const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_src; //!< elements in a full src chunk
    Index elems_dst; //!< elements in a full dst chunk
    int shift_src; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long scale_chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Chunk-aware scale: the SAME flat thread mapping (i = tid + bid*256)
//! and the SAME per-element expression (a single multiply, so there is
//! no FMA-contraction freedom) as cuda_kernel, with every operand address
//! resolved through its chunk table read straight from the device ring
//! (piece 9 / rule 10: no shared-memory staging, no table __syncthreads;
//! the table is a handful of L1-resident entries). The result is bitwise
//! cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha_, ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i < nelems)
    {
        long const f = long(i);
        Index const cs = Index(scale_chunk_index(f, t.elems_src, t.shift_src));
        Index const cd = Index(scale_chunk_index(f, t.elems_dst, t.shift_dst));
        t.dst[cd][f - cd * long(t.elems_dst)] =
            alpha * Y{t.src[cs][f - cs * long(t.elems_src)]};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        const T* const* src_chunks, Index nsrc, T* const* dst_chunks,
        Index ndst, Index elems_src, Index elems_dst)
    noexcept
{
    if(nelems <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_scale");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_scale");
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha,
            t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t *src, fp16_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t *src, fp64_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* src_chunks, Index nsrc,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src_chunks, Index nsrc,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src_chunks, Index nsrc,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src_chunks, Index nsrc,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

} // namespace nntile::kernel::scale
