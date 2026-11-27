/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/scale_inplace/cuda.cu
 * Scale inplace operation on buffers on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/scale_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::scale_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha_, T* data)
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      data[i] = alpha * data[i]
 *
 * @param[in] nelems: Size of the data tensor
 * @param[in] alpha_: Scalar multiplier for the data tensor
 * @param[inout] data: Destination of the scale inplace operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i < nelems)
    {
        data[i] = T{alpha * Y{data[i]}};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, T *data)
    noexcept
//! Set one buffer as a scaled version of another
/*! Performs the followin operation:
 *      dst[i] = alpha * src[i]
 *
 * @param[in] nelems: Size of the data tensor
 * @param[in] alpha: Scalar multiplier for the data tensor
 * @param[inout] data: Destination of the scale inplace operation. Input values are
 *      ignored, its content is overwritten on exit.
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, data);
}


template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, Scalar alpha_, ChunkMut<T> data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha{alpha_};
    if(i < nelems)
    {
        T *p = data.at(i);
        p[0] = T{alpha * Y{p[0]}};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems, Scalar alpha,
        T* const* data_chunks, Index ndata, Index elems_data, ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || ndata <= 0)
    {
        return;
    }
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha,
        chunk_mut<T>(data_chunks, ndata, elems_data, cls,
            "nntile_scale_inplace"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        fp32_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        fp16_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        fp64_t *data)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        fp32_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        fp64_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        bf16_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha,
        fp16_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

} // namespace nntile::kernel::scale_inplace
