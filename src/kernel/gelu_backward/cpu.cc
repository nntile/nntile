/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelu_backward/cpu.cc
 * Backward GeLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelu_backward/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::gelu_backward
{

template<typename T>
void cpu(Index nelems, const T *x_, const T *dy_, T *dx_)
    noexcept
//! Backward GeLU operation on CPU
/*! Does the following per-element operation:
 * dx[i] = dx[i] + dy[i]*GeLU'(x[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x_: Input value for forward GeLU
 * @params[in] dy_: Gradient over output of forward GeLU
 * @params[inout] dx_: Gradient over input of forward GeLU
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto x = reinterpret_cast<const Y *>(x_);
    auto dy = reinterpret_cast<const Y *>(dy_);
    auto dx = reinterpret_cast<Y *>(dx_);
    constexpr Y pi{3.141592653589793238462643383279502884L},
        one{1.0}, mone{-1.0}, pt5{0.5};
    const Y f1 = mone / std::sqrt(Y{2.0}), f2 = one / std::sqrt(2*pi);
    for(Index i = 0; i < nelems; ++i)
    {
        Y exp_x = std::exp(-pt5 * x[i] * x[i]);
        Y y = std::erfc(f1 * x[i]);
        dx[i] += (x[i]*f2*exp_x + pt5*y) * dy[i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *x, const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *x, const fp64_t *dy, fp64_t *dx)
    noexcept;

} // namespace nntile::kernel::gelu_backward
