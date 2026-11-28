/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/silu/cuda.cu
 * SiLU operation on CUDA
 *
 * @version 2.0.0
 * */

#include "nntile/kernel/silu/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::silu
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y one = Y{1.0};
    Y src_val{0.0};
    if(i < nelems)
    {
        src_val = Y{src[i]};
        dst[i] = T{src_val / (one + ::exp(-src_val))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src_, T *dst_)
    noexcept
//! SiLU operation on CUDA
/*! Does the following per-element operation:
 * dst[i] = src[i] * sigmoid(src[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input array
 * @params[out] dst_: Output array
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src_, dst_);
}


template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, ChunkConst<T> src, ChunkMut<T> dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    constexpr Y one = Y{1.0};
    if(i < nelems)
    {
        Y src_val = Y{src.at(i)[0]};
        dst.at(i)[0] = T{src_val / (one + ::exp(-src_val))};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        const T* const* src_chunks, Index nsrc,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
        chunk_const<T>(src_chunks, nsrc, elems_src, cls, "nntile_silu"),
        chunk_mut<T>(dst_chunks, ndst, elems_dst, cls, "nntile_silu"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *src,
        fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *src,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, const bf16_t *src,
        bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, const fp16_t *src,
        fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems,
        const fp32_t* const* src, Index nsrc, fp32_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index nelems,
        const fp64_t* const* src, Index nsrc, fp64_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index nelems,
        const bf16_t* const* src, Index nsrc, bf16_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index nelems,
        const fp16_t* const* src, Index nsrc, fp16_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::silu
