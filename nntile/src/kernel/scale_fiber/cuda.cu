/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/scale_fiber/cuda.cu
 * Per-element scaling of a tensor with a broadcasted fiber on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scale_fiber/cuda.hh"
#include <algorithm>
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::scale_fiber
{

template<typename T>
static __global__
void cuda_kernel(Index m, Index n, Index k, Index batch, Scalar alpha_, const T *src, T *dst)
//! Per-element scaling of a tensor with a broadcasted fiber on CUDA
/*! Performs the following operations:
 *      dst[i,l,j,b] = alpha*src[l,b]
 *
 * @param[in] m: Size of the first mode of dst tensor.
 * @param[in] n: Size of the last mode of dst tensor.
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors.
 * @param[in] batch: Size of the batch dimension.
 * @param[in] alpha_: Scalar factor for src.
 * @param[in] src: Input contiguous vector with k*batch elements.
 * @param[out] dst: Output contiguous m-by-k-by-n-by-batch array.
 * */
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i2 < k and i1 < n and i0 < m)
    {
        for(Index b = 0; b < batch; ++b)
        {
            // Value to scale and broadcast
            const Y src_val = alpha * Y{src[i2+b*k]};
            // Output fiber to be set
            T *dst_fiber = dst + ((i1+b*n)*k+i2)*m;
            // Set output value
            dst_fiber[i0] = T{src_val};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const T *src, T *dst)
    noexcept
//! Per-element scaling of a tensor with a broadcasted fiber on CUDA
/*! Performs the following operations:
 *      dst[i,l,j] = alpha*src[l]
 *
 * @param[in] m: Size of the first mode of dst tensor
 * @param[in] n: Size of the last mode of dst tensor
 * @param[in] k: Size of the middle mode of dst tensor and the only mode of src
 *      tensors
 * @param[in] batch: Size of the batch dimension
 * @param[in] alpha: Scalar factor for src
 * @param[in] src: Input contiguous vector with k elements
 * @param[out] dst: Output contiguous m-by-k-by-n array
 * */
{
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch, alpha, src, dst);
}

template<typename T>
static __global__
void chunked_cuda_kernel(Index m, Index n, Index k, Index batch,
        Scalar alpha_, ChunkConst<T> src, ChunkMut<T> dst)
{
    Index i2 = threadIdx.x + blockIdx.x*blockDim.x,
          i0 = threadIdx.y + blockIdx.y*blockDim.y,
          i1 = threadIdx.z + blockIdx.z*blockDim.z;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i2 < k and i1 < n and i0 < m)
    {
        for(Index b = 0; b < batch; ++b)
        {
            // Value to scale and broadcast
            const Y src_val = alpha * Y{src.at(i2+b*k)[0]};
            // Set output value
            dst.at(((i1+b*n)*k+i2)*m + i0)[0] = T{src_val};
        }
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha,
        const T* const* src_chunks, Index nsrc,
        T* const* dst_chunks, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept
{
    if(m <= 0 || n <= 0 || k <= 0 || batch <= 0 || nsrc <= 0 || ndst <= 0)
    {
        return;
    }
    // Both source and destination are Fortran-contiguous
    dim3 threads(std::min(int(k), 1024), std::min(int(m), 1),
            std::min(int(n), 1));
    dim3 blocks((k+threads.x-1)/threads.x, (m+threads.y-1)/threads.y,
            (n+threads.z-1)/threads.z);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(m, n, k, batch,
        alpha,
        chunk_const<T>(src_chunks, nsrc, elems_src, cls, "nntile_scale_fiber"),
        chunk_mut<T>(dst_chunks, ndst, elems_dst, cls, "nntile_scale_fiber"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const bf16_t *src, bf16_t *dst)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index m, Index n, Index k, Index batch,
        Scalar alpha, const fp16_t *src, fp16_t *dst)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp32_t* const* src, Index nsrc,
        fp32_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp64_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp64_t* const* src, Index nsrc,
        fp64_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<bf16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const bf16_t* const* src, Index nsrc,
        bf16_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;
template
void chunked<fp16_t>(cudaStream_t stream, Index m, Index n, Index k,
        Index batch, Scalar alpha, const fp16_t* const* src, Index nsrc,
        fp16_t* const* dst, Index ndst,
        Index elems_src, Index elems_dst, ChunkClass cls)
    noexcept;

} // namespace nntile::kernel::scale_fiber
