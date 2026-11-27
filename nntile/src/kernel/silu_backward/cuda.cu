/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_backward/cuda.cu
 * Backward SiLU operation on CUDA
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::silu_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    constexpr Y one{1.0};
    if(i < nelems)
    {
        Y x_val = Y{x[i]};
        Y dy_val = Y{dy[i]};
        Y sigma = one / (one + ::exp(-x_val));
        Y g = sigma * (one + x_val * (one - sigma));
        Y contrib = alpha_ * dy_val * g;
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
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *x_, const T *dy_,
        Scalar beta, T *dx_)
    noexcept
//! Backward SiLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = alpha * dy[i]*SiLU'(x[i]) + beta * dx[i]
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] alpha: Scalar multiplier for the gradient contribution
 * @params[in] x: Input value for forward SiLU
 * @params[in] dy: Gradient over output of forward SiLU
 * @params[in] beta: Scalar multiplier for the existing dx value
 * @params[inout] dx: Gradient over input of forward SiLU
 * */
{
    dim3 blocks((nelems+255)/256), threads(256);
    (cuda_kernel<T>)<<<blocks, threads, 0, stream>>>(nelems, alpha, x_, dy_, beta, dx_);
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

} // namespace nntile::kernel::silu_backward
