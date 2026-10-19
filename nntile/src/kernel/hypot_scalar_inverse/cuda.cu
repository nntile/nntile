/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot_scalar_inverse/cuda.cu
 * Inverse of a hypot operation of a buffer and a scalar on CUDA
 *
 * @version 1.3.0
 * */

#include "nntile/kernel/hypot_scalar_inverse/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::hypot_scalar_inverse
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar eps_, Scalar alpha_, T* dst)
//! Inverse of a hypot of a buffer and a scalar on CUDA
/*! Performs the following operation:
 *      dst[i] = 1.0 / hypot(alpha*dst[i], eps),
 * where alpha and eps are non-zero scalars.
 *
 * @param[in] nelems: Size of the dst tensor
 * @param[in] eps_: Scalar to be added to the hypot result
 * @param[in] alpha_: Scalar multiplier for the dst tensor
 * @param[inout] dst: Destination of the hypot operation
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y eps{eps_};
    if(i < nelems)
    {
        dst[i] = T{Y{1.0} / ::hypot(alpha*Y{dst[i]}, eps)};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha, T *dst)
    noexcept
//! Inverse of a hypot of a buffer and a scalar on CUDA
/*! Performs the following operation:
 *      dst[i] = 1.0 / hypot(alpha*dst[i], eps),
 * where alpha and eps are non-zero scalars.
 *
 * @param[in] nelems: Size of the dst tensor
 * @param[in] eps: Scalar to be added to the hypot result
 * @param[in] alpha: Scalar multiplier for the dst tensor
 * @param[inout] dst_: Destination of the hypot operation
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, eps, alpha, dst);
}

//! Chunk descriptors for the one-launch chunked hypot_scalar_inverse (see
//! cuda.hh).
template<typename T>
struct ChunkTables
{
    T *const* dst; //!< device-ring view of the dst pointer array
    Index ndst;
    Index elems_dst; //!< elements in a full dst chunk
    int shift_dst; //!< class shift (multi-chunk) or the 62 single-chunk sentinel
};

//! Chunk index of a flat element offset (power-of-two shift when possible).
static __device__ __forceinline__ long hsi_chunk_index(long f, Index elems,
    int shift) noexcept
{
    return f >> shift;
}

//! Chunk-aware hypot_scalar_inverse: the SAME flat thread mapping
//! (i = tid + bid*256) and the SAME per-element expression (copied
//! verbatim from cuda_kernel; hypot is a call, so there is no
//! FMA-contraction freedom) with the destination address resolved through
//! its chunk table read straight from the device ring (piece 9 / rule
//! 10: no staging, no table __syncthreads). The result is bitwise
//! cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar eps_, Scalar alpha_,
        ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    const Y eps{eps_};
    if(i < nelems)
    {
        long const f = long(i);
        Index const c = Index(hsi_chunk_index(f, t.elems_dst, t.shift_dst));
        T *p = t.dst[c] + (f - c * long(t.elems_dst));
        p[0] = T{Y{1.0} / ::hypot(alpha*Y{p[0]}, eps)};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha,
        T* const* dst_chunks, Index ndst, Index elems_dst)
    noexcept
{
    if(nelems <= 0 || ndst <= 0)
    {
        return;
    }
    ChunkTables<T> t;
    t.dst = dst_chunks;
    t.ndst = ndst;
    t.elems_dst = elems_dst;
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_hypot_scalar_inverse");
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(
        nelems, eps, alpha, t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha,
        fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha,
        fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha,
        bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar eps, Scalar alpha,
        fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar eps,
        Scalar alpha, fp32_t* const* dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar eps,
        Scalar alpha, fp64_t* const* dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar eps,
        Scalar alpha, bf16_t* const* dst_chunks, Index ndst, Index elems_dst)
    noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar eps,
        Scalar alpha, fp16_t* const* dst_chunks, Index ndst, Index elems_dst)
    noexcept;

} // namespace nntile::kernel::hypot_scalar_inverse
