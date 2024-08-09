/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelu_backward/cuda.cu
 * Backward GeLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelu_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelu_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *x, const T *dy, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    constexpr T pi = 3.141592653589793238462643383279502884L,
        one = 1, mone = -1, pt5 = 0.5;
    const T f1 = mone / std::sqrt(T{2.0}), f2 = one / std::sqrt(2*pi);
    if(i < nelems)
    {
        // T z = x[i];
        T exp_x = std::exp(-pt5 * x[i] * x[i]);
        T y = erfc(f1 * x[i]);
        dx[i] += (x[i]*f2*exp_x + pt5*y) * dy[i];
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x_, const T *dy_, T *dx_)
    noexcept
//! Backward GeLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = dx[i] + dy[i]*GeLU'(x[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x_: Input value for forward GeLU
 * @params[in] dy_: Gradient over output of forward GeLU
 * @params[out] dx_: Gradient over input of forward GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    using Y = typename CUDAComputeType<T>::value;
    auto x = reinterpret_cast<const Y *>(x_);
    auto dy = reinterpret_cast<const Y *>(dy_);
    auto dx = reinterpret_cast<Y *>(dx_);
    (cuda_kernel<Y>)<<<blocks, threads, 0, stream>>>(nelems, x, dy, dx);
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

} // namespace nntile::kernel::gelu_backward
