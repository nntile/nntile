/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/hypot/cuda.cu
 * hypot operation on buffers on CUDA
 *
 * @version 2.0.0
 * */

#include "nntile/kernel/hypot/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::hypot
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T* src1, Scalar beta, const T* src2, T* dst)
//! Generic implementation of the hypot operation on CUDA
/*! @copydoc nntile::kernel::hypot::cuda
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha_{alpha};
    const Y beta_{beta};
    if(i < nelems)
    {
        const Y src1_val = static_cast<Y>(src1[i]);
        const Y src2_val = static_cast<Y>(src2[i]);
        dst[i] = static_cast<T>(std::hypot(alpha*src1_val, beta*src2_val));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T* src1, Scalar beta, const T* src2, T* dst)
    noexcept
//! Hypothenuse of two buffers with optional scaling on CUDA
/*! Performs the following operation:
 * dst[i] = hypot(alpha*src1[i], beta*src2[i])
 *
 * This function reads both src1 and src2 even if alpha or beta is zero.
 * If alpha is zero and src1[i] is NaN, then dst[i] will be NaN.
 * If beta is zero and src2[i] is NaN, then dst[i] will be NaN.
 * If such behaviour is not desired, then in a case of alpha or beta being
 * zero, use nntile::kernel::scale instead.
 * If both alpha and beta are zero, then use nntile::kernel::clear instead.
 *
 * @see nntile::kernel::scale
 * @see nntile::kernel::clear
 *
 * @param[in] nelems: Size of the src and dst tensors
 * @param[in] alpha: Scalar multiplier for the src1 tensor
 * @param[in] src1: First source tensor
 * @param[in] beta: Scalar multiplier for the src2 tensor
 * @param[in] src2: Second source tensor
 * @param[out] dst: Destination tensor
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, src1, beta, src2, dst);
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha, ChunkConst<T> src1,
        Scalar beta, ChunkConst<T> src2, ChunkMut<T> dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    if(i < nelems)
    {
        const Y src1_val = static_cast<Y>(src1.at(i)[0]);
        const Y src2_val = static_cast<Y>(src2.at(i)[0]);
        dst.at(i)[0] = static_cast<T>(std::hypot(alpha*src1_val,
                    beta*src2_val));
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        Scalar alpha,
        const T* const* src1_chunks, Index nsrc1,
        Scalar beta,
        const T* const* src2_chunks, Index nsrc2,
        T* const* dst_chunks, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || nsrc1 <= 0 || nsrc2 <= 0 || ndst <= 0)
    {
        return;
    }
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha,
        chunk_const<T>(src1_chunks, nsrc1, elems_src1, cls, "nntile_hypot"),
        beta,
        chunk_const<T>(src2_chunks, nsrc2, elems_src2, cls, "nntile_hypot"),
        chunk_mut<T>(dst_chunks, ndst, elems_dst, cls, "nntile_hypot"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t* src1,
        Scalar beta, const fp32_t* src2, fp32_t* dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t* src1,
        Scalar beta, const fp64_t* src2, fp64_t* dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t* src1,
        Scalar beta, const bf16_t* src2, bf16_t* dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp16_t* src1,
        Scalar beta, const fp16_t* src2, fp16_t* dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp32_t* const* src1, Index nsrc1, Scalar beta,
        const fp32_t* const* src2, Index nsrc2, fp32_t* const* dst, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp64_t* const* src1, Index nsrc1, Scalar beta,
        const fp64_t* const* src2, Index nsrc2, fp64_t* const* dst, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const bf16_t* const* src1, Index nsrc1, Scalar beta,
        const bf16_t* const* src2, Index nsrc2, bf16_t* const* dst, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        const fp16_t* const* src1, Index nsrc1, Scalar beta,
        const fp16_t* const* src2, Index nsrc2, fp16_t* const* dst, Index ndst,
        Index elems_src1, Index elems_src2, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::hypot
