/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/rope_backward/cuda.cu
 * Backward for Rotary Positional Embedding
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/rope_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::rope_backward
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, const T *sin, const T *cos, const T *src,
        T *dst)
/*! Change provided 2-by-m-by-n src tensor and write result into dst tensor
 *  sin, cos are tensors of shape (m). Each column holds sines and cosines.
 *  TODO: describe math here
 *
 * @param[in] m: Size of sin and cos tensors
 * @param[in] n: Size of the second mode of src and dst tensors
 * @param[in] sin: Input sine tensor
 * @param[in] cos: Input cosine tensor
 * @param[in] src: Gradient over output of forward RoPE
 * @param[out] dst: Gradient over input of forward RoPE
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < m*n)
    {
        using Y = typename T::repr_t;
        int j = i % m;
        Y c{cos[j]}, s{sin[j]};
        Y a{src[2*i]}, b{src[2*i+1]};
        dst[2*i] = static_cast<T>(c*a + s*b);
        dst[2*i+1] = static_cast<T>(-s*a + c*b);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, const T *sin, const T *cos,
        const T *src, T *dst)
    noexcept
{
    dim3 blocks((m*n+255)/256), threads(256);
    cuda_kernel<T><<<blocks, threads, 0, stream>>>(m, n, sin, cos, src, dst);
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index m, Index n, ChunkConst<T> sin,
        ChunkConst<T> cos, ChunkConst<T> src, ChunkMut<T> dst)
{
    Index i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < m*n)
    {
        using Y = typename T::repr_t;
        Index j = i % m;
        Y c{cos.at(j)[0]}, s{sin.at(j)[0]};
        Y a{src.at(2*i)[0]}, b{src.at(2*i+1)[0]};
        dst.at(2*i)[0] = static_cast<T>(c*a + s*b);
        dst.at(2*i+1)[0] = static_cast<T>(-s*a + c*b);
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n,
        const T* const* sin_chunks, Index nsin,
        const T* const* cos_chunks, Index ncos,
        const T* const* dy_chunks, Index ndy,
        T* const* dx_chunks, Index ndx,
        Index elems_sin, Index elems_cos, Index elems_dy, Index elems_dx,
        ChunkClass cls)
    noexcept
{
    if(nsin <= 0 || ncos <= 0 || ndy <= 0 || ndx <= 0)
    {
        return;
    }
    dim3 blocks((m*n+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n,
        chunk_const<T>(sin_chunks, nsin, elems_sin, cls, "nntile_rope_backward"),
        chunk_const<T>(cos_chunks, ncos, elems_cos, cls, "nntile_rope_backward"),
        chunk_const<T>(dy_chunks, ndy, elems_dy, cls, "nntile_rope_backward"),
        chunk_mut<T>(dx_chunks, ndx, elems_dx, cls, "nntile_rope_backward"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, const fp32_t *sin,
        const fp32_t *cos, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, const fp64_t *sin,
        const fp64_t *cos, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, const fp16_t *sin,
        const fp16_t *cos, const fp16_t *src, fp16_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, const bf16_t *sin,
        const bf16_t *cos, const bf16_t *src, bf16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n,
        const fp32_t* const* sin_chunks, Index nsin,
        const fp32_t* const* cos_chunks, Index ncos,
        const fp32_t* const* dy_chunks, Index ndy,
        fp32_t* const* dx_chunks, Index ndx, Index elems_sin,
        Index elems_cos, Index elems_dy, Index elems_dx, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n,
        const fp64_t* const* sin_chunks, Index nsin,
        const fp64_t* const* cos_chunks, Index ncos,
        const fp64_t* const* dy_chunks, Index ndy,
        fp64_t* const* dx_chunks, Index ndx, Index elems_sin,
        Index elems_cos, Index elems_dy, Index elems_dx, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n,
        const bf16_t* const* sin_chunks, Index nsin,
        const bf16_t* const* cos_chunks, Index ncos,
        const bf16_t* const* dy_chunks, Index ndy,
        bf16_t* const* dx_chunks, Index ndx, Index elems_sin,
        Index elems_cos, Index elems_dy, Index elems_dx, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n,
        const fp16_t* const* sin_chunks, Index nsin,
        const fp16_t* const* cos_chunks, Index ncos,
        const fp16_t* const* dy_chunks, Index ndy,
        fp16_t* const* dx_chunks, Index ndx, Index elems_sin,
        Index elems_cos, Index elems_dy, Index elems_dx, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::rope_backward
