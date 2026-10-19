/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh/cuda.cu
 * Approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.3.0
 * */

#include "nntile/kernel/gelutanh/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *src, T *dst)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    // Constants
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1;
    if(i < nelems)
    {
        Y z = Y{src[i]};
        Y y = z * (f3 + f4*z*z);
        dst[i] = T{z / (one + ::exp(y))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *src, T *dst)
    noexcept
//! Approximate GeLU operation on CUDA
/*! Applies the following approximation of the GeLU function:
 * GeLU(z) \approx 0.5z(1+tanh(sqrt(2/pi)(z+0.044715z^3))),
 * which is actually implemented as
 * GeLU(z) \approx z/(1+exp(-2sqrt(2/pi)z(1+0.044715z^2)))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src: Input buffer to apply GeLU
 * @params[out] dst: Output buffer to apply GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, src, dst);
}

//! Chunk descriptors for the one-launch chunked gelutanh (see cuda.hh).
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
static __device__ __forceinline__ long gelutanh_chunk_index(long f,
    Index elems, int shift) noexcept
{
    return f >> shift;
}

//! Chunk-aware gelutanh: the SAME flat thread mapping (i = tid + bid*256)
//! and the SAME per-element expression (copied verbatim from cuda_kernel,
//! including the per-thread constant derivation, so nvcc's contraction
//! decisions are the classic kernel's) with every operand address resolved
//! through its chunk table read straight from the device ring (piece
//! 9 / rule 10: no staging, no table __syncthreads). The result is
//! bitwise cuda()'s.
template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, ChunkTables<T> t)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    // Constants
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1;
    if(i < nelems)
    {
        long const f = long(i);
        Index const cs = Index(gelutanh_chunk_index(f, t.elems_src, t.shift_src));
        Index const cd = Index(gelutanh_chunk_index(f, t.elems_dst, t.shift_dst));
        Y z = Y{t.src[cs][f - cs * long(t.elems_src)]};
        Y y = z * (f3 + f4*z*z);
        t.dst[cd][f - cd * long(t.elems_dst)] = T{z / (one + ::exp(y))};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
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
    t.shift_src = chunk_shift_of<T>(t.nsrc, elems_src, "nntile_gelutanh");
    t.shift_dst = chunk_shift_of<T>(t.ndst, elems_dst, "nntile_gelutanh");
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads,
        0, stream>>>(nelems, t);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t * src,
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
void cuda<fp16_t>(cudaStream_t stream, Index nelems, const fp16_t * src,
        fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems,
        const fp32_t* const* src_chunks, Index nsrc,
        fp32_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems,
        const fp64_t* const* src_chunks, Index nsrc,
        fp64_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems,
        const bf16_t* const* src_chunks, Index nsrc,
        bf16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems,
        const fp16_t* const* src_chunks, Index nsrc,
        fp16_t* const* dst_chunks, Index ndst, Index elems_src,
        Index elems_dst) noexcept;

} // namespace nntile::kernel::gelutanh
