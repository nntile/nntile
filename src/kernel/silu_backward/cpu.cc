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
void cpu(Index nelems, const T *x, const T *dy, T *dx)
    noexcept
//! Backward SiLU operation on CPU
/*! Does the following per-element operation:
 * dx[i] = dx[i] + dy[i]*SiLU'(x[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x: Input value for forward SiLU
 * @params[in] dy: Gradient over output of forward SiLU
 * @params[inout] dx: Gradient over input of forward SiLU
 * */
{
    using Y = typename T::repr_t;
    Y x_val{0.0};
    Y dx_val{0.0};
    Y dy_val{0.0};
    Y sigma{0.0};
    constexpr Y one{1.};
    for(Index i = 0; i < nelems; ++i)
    {
        x_val = static_cast<Y>(x[i]);
        sigma = one / (one + std::exp(-x_val));
        dx_val = static_cast<Y>(dx[i]);
        dy_val = static_cast<Y>(dy[i]);
        dy_val *= sigma * (one + x_val * (one - sigma));
        dx[i] = static_cast<T>(dx_val + dy_val);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *x, const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *x, const fp64_t *dy, fp64_t *dx)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *x, const bf16_t *dy, bf16_t *dx)
    noexcept;

template
void cpu<fp32_fast_tf32_t>(Index nelems, const fp32_fast_tf32_t *x, const fp32_fast_tf32_t *dy,
                           fp32_fast_tf32_t *dx)
    noexcept;

} // namespace nntile::kernel::silu_backward
