/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
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
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    constexpr T pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = T{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const T sqrt_pi = sqrt(pi), sqrt_2 = sqrt(T{2}),
        f2 = sqrt_2/sqrt_pi, f3 = -T{2}*f2, f4 = f3*f1;
    if(i < nelems)
    {
        T z = data[i];
        T y = z * (f3 + f4*z*z);
        data[i] = z / (one+::exp(y));
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data_)
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
    using Y = typename CUDAComputeType<T>::value;
    auto data = reinterpret_cast<Y *>(data_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(nelems, data);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, fp32_t *data)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::gelutanh_inplace
