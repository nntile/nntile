/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/gelu/cuda.cu
 * GeLU operation on a buffer on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelu/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelu
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y mone = -1, pt5 = 0.5;
    const Y f1 = mone / sqrt(Y{2.0});

    if(i < nelems)
    {
        Y z = Y{src[i]};
        Y y = ::erfc(f1 * z);
        dst[i] = T{pt5 * z * y};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! GeLU operation performed on CUDA
/*! Uses erfc() function, which implements the following:
 * GeLU(z) = 0.5 z erfc(-z/sqrt(2))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input buffer to apply GeLU
 * @params[out] dst: Output buffer to apply GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}


//! Chunk descriptors for the one-launch chunked gelu.
template<typename T>
struct GeluChunkTables
{
    const T* const* src;
    Index nsrc;
    T* const* dst;
    Index ndst;
    Index elems_src;
    Index elems_dst;
    int shift_src;
    int shift_dst;
    long mask_src;
    long mask_dst;
    long nelems;
};

template<typename T>
static __global__
void chunked_cuda_kernel(GeluChunkTables<T> t)
{
    long const i = long(blockIdx.x)*blockDim.x + threadIdx.x;
    using Y = typename T::repr_t;
    if(i < t.nelems)
    {
        long const sc = i >> t.shift_src;
        long const dc = i >> t.shift_dst;
        const T *src = t.src[sc] + (i & t.mask_src);
        T *dst = t.dst[dc] + (i & t.mask_dst);
        Y z = Y{*src};
        constexpr Y mone = -1, pt5 = 0.5;
        Y const f1 = mone / sqrt(Y{2.0});
        Y y = ::erfc(f1 * z);
        *dst = T{pt5 * z * y};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        const T* const* src_chunks, Index nsrc,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst,
        ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    GeluChunkTables<T> t;
    t.src = src_chunks;
    t.nsrc = nsrc;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_src = elems_src;
    t.elems_dst = elems_dst;
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, cls, "nntile_gelu");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, cls, "nntile_gelu");
    t.mask_src = static_cast<long>(chunk_mask_of<T>(t.nsrc, cls));
    t.mask_dst = static_cast<long>(chunk_mask_of<T>(t.ndst, cls));
    t.nelems = long(nelems);
    dim3 threads(256);
    dim3 blocks((nelems+255)/256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, const fp16_t *src, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems,
        const fp32_t* const* src_chunks, Index nsrc,
        fp32_t* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems,
        const fp64_t* const* src_chunks, Index nsrc,
        fp64_t* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems,
        const bf16_t* const* src_chunks, Index nsrc,
        bf16_t* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems,
        const fp16_t* const* src_chunks, Index nsrc,
        fp16_t* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::gelu
