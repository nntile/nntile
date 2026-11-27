/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_backward/cpu.cc
 * Backward of approximate GeLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh_backward/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::gelutanh_backward
{

template<typename T>
void cpu(Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
    noexcept
//! Backward of approximate GeLU operation on CPU
/*! Applies the following derivative of approximation of the GeLU function:
 * dx[i] = alpha * dy[i]*GeLUtanh'(x[i]) + beta * dx[i]
 * GeLUtanh'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] alpha: Scalar multiplier for the gradient contribution
 * @params[in] x: Input value for forward approximate GeLU
 * @params[in] dy: Gradient over output of forward approximate GeLU
 * @params[in] beta: Scalar multiplier for the existing dx value
 * @params[inout] dx: Gradient over input of forward approximate GeLU
 * */
{
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    // Constants
    constexpr Y pi{3.141592653589793238462643383279502884L},
        one{1.0}, pt5{0.5}, f1{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    static const Y sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2.0}*f2, f4 = f3*f1, f5 = Y{3.0}*f4;
    for(Index i = 0; i < nelems; ++i)
    {
        Y x_val = Y{x[i]};
        Y z2 = x_val * x_val;
        Y y1 = x_val * (f3 + f4*z2);
        Y y2 = x_val * (f3 + f5*z2);
        Y expy1 = std::exp(y1);
        if(not std::isinf(expy1))
        {
            Y inv_expy1p1 = one / (expy1 + one);
            Y g = (one-y2*(one-inv_expy1p1)) * inv_expy1p1;
            Y contrib = alpha_ * g * Y{dy[i]};
            if(beta == 0.0)
            {
                dx[i] = static_cast<T>(contrib);
            }
            else
            {
                dx[i] = static_cast<T>(contrib + beta_ * Y{dx[i]});
            }
        }
        else
        {
            if(beta == 0.0)
            {
                dx[i] = static_cast<T>(Y{0.0});
            }
            else if(beta != 1.0)
            {
                dx[i] = static_cast<T>(beta_ * Y{dx[i]});
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, Scalar alpha, const fp32_t *x, const fp32_t *dy,
        Scalar beta, fp32_t *dx)
    noexcept;

template
void cpu<fp64_t>(Index nelems, Scalar alpha, const fp64_t *x, const fp64_t *dy,
        Scalar beta, fp64_t *dx)
    noexcept;

template
void cpu<bf16_t>(Index nelems, Scalar alpha, const bf16_t *x, const bf16_t *dy,
        Scalar beta, bf16_t *dx)
    noexcept;

template
void cpu<fp16_t>(Index nelems, Scalar alpha, const fp16_t *x, const fp16_t *dy,
        Scalar beta, fp16_t *dx)
    noexcept;

} // namespace nntile::kernel::gelutanh_backward
