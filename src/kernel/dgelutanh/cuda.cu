/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/dgelutanh/cuda.cu
 * Derivative of approximate GeLU operation on CUDA based on tanh function
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/dgelutanh/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::dgelutanh
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, T *data)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    constexpr T pi = 3.141592653589793238462643383279502884L,
        zero = 0, one = 1, f1 = T{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const T sqrt_pi = sqrt(pi), sqrt_2 = sqrt(T{2}),
        f2 = sqrt_2/sqrt_pi, f3 = -T{2}*f2, f4 = f3*f1, f5 = T{3}*f4;
    if(i < nelems)
    {
        T z = data[i];
        T z2 = z * z;
        T y1 = z * (f3 + f4*z2);
        T y2 = z * (f3 + f5*z2);
        T expy1 = exp(y1);
        if(isinf(expy1))
        {
            data[i] = zero;
        }
        else
        {
            T inv_expy1p1 = one / (expy1 + one);
            data[i] = (one-y2*(one-inv_expy1p1)) * inv_expy1p1;
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, T *data_)
    noexcept
//! Derivative of approximate GeLU operation on CUDA
/*! Applies the following derivative of approximation of the GeLU function:
 * GeLU(z) \approx AGeLU(z)
 * f(z) = -2 sqrt(2/pi) z (1+0.044715z^2)
 * AGeLU(z) = z / (1+exp(f(z))
 * AGeLU'(z) = 1/(1+exp(f(z)) - (zf'(z)exp(f(z)))/(1+exp(f(z)))^2
 * AGeLU'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 * zf'(z) = -2 sqrt(2/pi) z (1+3*0.044715z^2)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply derivative of approximate GeLU
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

} // namespace nntile::kernel::dgelutanh
