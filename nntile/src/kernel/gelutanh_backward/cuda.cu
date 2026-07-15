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
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh_backward/cuda.hh"
#include "nntile/kernel/cuda.hh"

namespace nntile::kernel::gelutanh_backward
{

template<typename T>
static __global__
void cuda_kernel(Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    // Constants
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    constexpr Y pi = 3.141592653589793238462643383279502884L,
        one = 1, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    const Y sqrt_pi = sqrt(pi), sqrt_2 = sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2}*f2, f4 = f3*f1, f5 = Y{3}*f4;
    if(i < nelems)
    {
        Y z = Y{x[i]};
        Y z2 = z * z;
        Y y1 = z * (f3 + f4*z2);
        Y y2 = z * (f3 + f5*z2);
        Y expy1 = exp(y1);
        if(not isinf(expy1))
        {
            Y inv_expy1p1 = one / (expy1 + one);
            Y g = (one-y2*(one-inv_expy1p1)) * inv_expy1p1;
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
        else
        {
            if(beta == 0.0)
            {
                dx[i] = T{Y{0.0}};
            }
            else if(beta != 1.0)
            {
                dx[i] = T{beta_ * Y{dx[i]}};
            }
        }
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index nelems, Scalar alpha, const T *x, const T *dy,
        Scalar beta, T *dx)
    noexcept
//! Backward approximate GeLU operation on CUDA
/*! Does the following per-element operation:
 * dx[i] = alpha * dy[i]*GeLUtanh'(x[i]) + beta * dx[i]
 * GeLUtanh'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] alpha: Scalar multiplier for the gradient contribution
 * @params[in] x: Input value for forward GeLU
 * @params[in] dy: Gradient over output of forward GeLU
 * @params[in] beta: Scalar multiplier for the existing dx value
 * @params[inout] dx: Gradient over input of forward GeLU
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

} // namespace nntile::kernel::gelutanh_backward
