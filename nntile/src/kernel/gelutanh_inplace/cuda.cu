/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/gelutanh_inplace/cuda.cu
 * Approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh_inplace/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh_inplace
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    using Y = typename T::repr_t;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1;
    if(i < nelems)
    {
        Y z = static_cast<Y>(data[i]);
        Y y = z * (f3 + f4*z*z);
        data[i] = T{z / (one+::exp(y))};
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data)
    noexcept
//! Approximate GeLU operation on CUDA
/*! Applies the following approximation of the GeLU function:
 * GeLU(z) \approx 0.5z(1+tanh(sqrt(2/pi)(z+0.044715z^3))),
 * which is actually implemented as
 * GeLU(z) \approx z/(1+exp(-2sqrt(2/pi)z(1+0.044715z^2)))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, data);
}


template<typename T>
static __global__
void chunked_cuda_kernel(Index nelems, ChunkMut<T> data)
{
    using Y = typename T::repr_t;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1;
    if(i < nelems)
    {
        T *p = data.at(i);
        Y z = static_cast<Y>(p[0]);
        Y y = z * (f3 + f4*z*z);
        p[0] = T{z / (one+::exp(y))};
    }
}

template<typename T>
void chunked(cudaStream_t stream, Index nelems,
        T* const* data_chunks, Index ndata, Index elems_data, ChunkClass cls)
    noexcept
{
    if(nelems <= 0 || ndata <= 0)
    {
        return;
    }
    dim3 blocks((nelems+255)/256), threads(256);
    (chunked_cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems,
        chunk_mut<T>(data_chunks, ndata, elems_data, cls,
            "nntile_gelutanh_inplace"));
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, bf16_t *data)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, fp16_t *data)
    noexcept;

template
void chunked<fp32_t>(cudaStream_t stream, Index nelems,
        fp32_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

template
void chunked<fp64_t>(cudaStream_t stream, Index nelems,
        fp64_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

template
void chunked<bf16_t>(cudaStream_t stream, Index nelems,
        bf16_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

template
void chunked<fp16_t>(cudaStream_t stream, Index nelems,
        fp16_t* const* data_chunks, Index ndata, Index elems_data,
        ChunkClass cls) noexcept;

} // namespace nntile::kernel::gelutanh_inplace
