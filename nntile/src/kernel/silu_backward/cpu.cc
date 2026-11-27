/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_backward/cpu.cc
 * Backward SiLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_backward/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::silu_backward
{

template<typename T>
void cpu(Index nelems, Scalar alpha, const T *x, const T *dy, Scalar beta, T *dx)
    noexcept
//! Backward SiLU operation on CPU
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
    using Y = typename T::repr_t;
    const Y alpha_{alpha}, beta_{beta};
    Y x_val{0.0};
    Y dy_val{0.0};
    Y sigma{0.0};
    constexpr Y one{1.};
    for(Index i = 0; i < nelems; ++i)
    {
        x_val = static_cast<Y>(x[i]);
        dy_val = static_cast<Y>(dy[i]);
        sigma = one / (one + std::exp(-x_val));
        Y g = sigma * (one + x_val * (one - sigma));
        Y contrib = alpha_ * dy_val * g;
        if(beta == 0.0)
        {
            dx[i] = static_cast<T>(contrib);
        }
        else
        {
            dx[i] = static_cast<T>(contrib + beta_ * static_cast<Y>(dx[i]));
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

} // namespace nntile::kernel::silu_backward
