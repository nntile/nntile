/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/hypot_inplace/cuda.cu
 * hypot_inplace operation on buffers on CUDA
 *
 * @version 2.0.0
 * */

#include "nntile/kernel/hypot_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::hypot_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T* src, Scalar beta, T* dst)
//! Generic implementation of the hypot_inplace operation on CUDA
/*! @copydoc nntile::kernel::hypot_inplace::cuda
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    if(i < nelems)
    {
        const Y src_val = static_cast<Y>(src[i]);
        const Y dst_val = static_cast<Y>(dst[i]);
        dst[i] = static_cast<T>(std::hypot(alpha_*src_val, beta_*dst_val));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *src,
        Scalar beta, T *dst)
    noexcept
//! Hypothenuse of two buffers with optional scaling inplace on CUDA
/*! Performs the following operation:
 * dst[i] = hypot(alpha*src[i], beta*dst[i])
 *
 * This function reads both src and dst even if alpha or beta is zero.
 * If alpha is zero and src[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and dst[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha being zero,
 * use nntile::kernel::scale_inplace instead, and in a case of beta being
 * zero, use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale_inplace
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src tensor
 * @param[in] src: Source tensor
 * @param[in] beta: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the hypot_inplace operation
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src,
            beta, dst);
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha, ChunkConst<T> src,
        Scalar beta, ChunkMut<T> dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    if(i < nelems)
    {
        const Y src_val = static_cast<Y>(src.at(i)[0]);
        const Y dst_val = static_cast<Y>(dst.at(i)[0]);
        dst.at(i)[0] = static_cast<T>(std::hypot(alpha_*src_val,
                    beta_*dst_val));
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        Scalar alpha,
        const T* const* src_chunks, Index nsrc,
        Scalar beta,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha,
        chunk_const<T>(src_chunks, nsrc, elems_src, cls, "nntile_hypot_inplace"),
        beta,
        chunk_mut<T>(dst_chunks, ndst, elems_dst, cls, "nntile_hypot_inplace"));
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
        const fp32_t* const* src, Index nsrc, Scalar beta,
        fp32_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src, Index nsrc, Scalar beta,
        fp64_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src, Index nsrc, Scalar beta,
        bf16_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src, Index nsrc, Scalar beta,
        fp16_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::hypot_inplace
