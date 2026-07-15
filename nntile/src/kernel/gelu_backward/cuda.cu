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
void cuda_kernel(Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, mone = -1, pt5 = 0.5;
    const Y f1 = mone / std::sqrt(Y{2.0}), f2 = one / std::sqrt(2*pi);
    if(i < nelems)
    {
        Y x_val = Y{x[i]};
        Y exp_x = std::exp(-pt5 * x_val * x_val);
        Y y = erfc(f1 * x_val);
        Y g = x_val*f2*exp_x + pt5*y;
        Y contrib = alpha_ * g * Y{dy[i]};
        if(beta == 0.0)
        {
            dx[i] = T{contrib};
        }
        else
        {
            dx[i] = T{contrib + beta_ * Y{dx[i]}};
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *x, const T *dy,
        Scalar beta, T *dx)
    noexcept
//! Backward GeLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = alpha * dy[i]*GeLU'(x[i]) + beta * dx[i]
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] alpha: Scalar multiplier for the gradient contribution
 * @params[in] x: Input value for forward GeLU
 * @params[in] dy: Gradient over output of forward GeLU
 * @params[in] beta: Scalar multiplier for the existing dx value
 * @params[out] dx: Gradient over input of forward GeLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, x, dy, beta, dx);
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp32_t *x,
        const fp32_t *dy, Scalar beta, fp32_t *dx)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp64_t *x,
        const fp64_t *dy, Scalar beta, fp64_t *dx)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const bf16_t *x,
        const bf16_t *dy, Scalar beta, bf16_t *dx)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index nelems, Scalar alpha, const fp16_t *x,
        const fp16_t *dy, Scalar beta, fp16_t *dx)
    noexcept;

} // namespace nntile::kernel::gelu_backward
