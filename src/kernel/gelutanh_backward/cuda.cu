/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_backward/cuda.cu
 * Backward approximate GeLU operation on CUDA
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/gelutanh_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *x, const T *dy, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    constexpr T pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = T{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const T sqrt_pi = sqrt(pi), sqrt_2 = sqrt(T{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -T{2}*f2, f4 = f3*f1, f5 = T{3}*f4;
    if(i < nelems)
    {
        T z = x[i];
        T z2 = z * z;
        T y1 = z * (f3 + f4*z2);
        T y2 = z * (f3 + f5*z2);
        T expy1 = exp(y1);
        if(not isinf(expy1))
        {
            T inv_expy1p1 = one / (expy1 + one);
            dx[i] += (one-y2*(one-inv_expy1p1)) * inv_expy1p1 * dy[i];
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x_, const T *dy_, T *dx_)
    noexcept
//! Backward approximate GeLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = dx[i] + dy[i]*GeLUtanh'(x[i])
 * GeLUtanh'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x_: Input value for forward GeLU
 * @params[in] dy_: Gradient over output of forward GeLU
 * @params[inout] dx_: Gradient over input of forward GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    using Y = typename CUDAComputeType<T>::value;
    auto x = reinterpret_cast<const Y *>(x_);
    auto dy = reinterpret_cast<const Y *>(dy_);
    auto dx = reinterpret_cast<Y *>(dx_);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, x, dy, dx);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, const fp32_t *x,
        const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, const fp64_t *x,
        const fp64_t *dy, fp64_t *dx)
    noexcept;

} // namespace nntile::kernel::gelutanh_backward
