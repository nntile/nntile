/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_backward/cuda.cu
 * Backward approximate GeLU operation on CUDA
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-04-05
 * */

#include "nntile/kernel/gelutanh_backward/cuda.hh"

namespace nntile
{
namespace kernel
{
namespace gelutanh_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, const T *x, const T *dy, T *dx)
{
    int start = threadIdx.x + blockIdx.x*blockDim.x,
        step = blockDim.x * gridDim.x;
    // Constants
    constexpr T pi = 3.141592653589793238462643383279502884L,
        zero = 0, one = 1, f1 = T{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const T sqrt_pi = sqrt(pi), sqrt_2 = sqrt(T{2}),
        f2 = sqrt_2/sqrt_pi, f3 = -T{2}*f2, f4 = f3*f1, f5 = T{3}*f4;
    for(Index i = start; i < nelems; i += step)
    {
        T z = x[i];
        T z2 = z * z;
        T y1 = z * (f3 + f4*z2);
        T y2 = z * (f3 + f5*z2);
        T expy1 = exp(y1);
        if(isinf(expy1))
        {
            dx[i] = zero;
        }
        else
        {
            T inv_expy1p1 = one / (expy1 + one);
            dx[i] = (one-y2*(one-inv_expy1p1)) * inv_expy1p1 * dy[i];
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, const T *x, const T *dy, T *dx)
    noexcept
//! Backward approximate GeLU operation on CUDA
/*! Does the following per-element operation:
 * backward_GeLU(x, dy) = GeLU'(x) * dy elementwise
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x: Input value for forward GeLU
 * @params[in] dy: Gradient over output of forward GeLU
 * @params[out] dx: Gradient over input of forward GeLU
 * */
{
    dim3 blocks(256), threads(32);
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

} // namespace gelutanh_backward
} // namespace kernel
} // namespace nntile

