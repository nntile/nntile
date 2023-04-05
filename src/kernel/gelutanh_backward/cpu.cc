/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_backward/cpu.cc
 * Derivative of approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-04-05
 * */

#include "nntile/kernel/gelutanh_backward/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace gelutanh_backward
{

template<typename T>
void cpu(Index nelems, const T *x, const T *dy, T *dx)
    noexcept
//! Derivative of approximate GeLU backward operation on CPU
/*! Applies the following derivative of approximation of the GeLU function:
 * GeLU(z) \approx AGeLU(z)
 * f(z) = -2 sqrt(2/pi) z (1+0.044715z^2)
 * AGeLU(z) = z / (1+exp(f(z))
 * AGeLU'(z) = 1/(1+exp(f(z)) - (zf'(z)exp(f(z)))/(1+exp(f(z)))^2
 * AGeLU'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 * zf'(z) = -2 sqrt(2/pi) z (1+3*0.044715z^2)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] x: Input value for forward approximate GeLU
 * @params[in] dy: Gradient over output of forward approximate GeLU
 * @params[out] dx: Gradient over input of forward approximate GeLU
 * */
{
    // Constants
    constexpr T pi = 3.141592653589793238462643383279502884L,
        zero = 0, one = 1, pt5 = 0.5, f1 = T{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    static const T sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(T{2}),
        f2 = sqrt_2/sqrt_pi, f3 = -T{2}*f2, f4 = f3*f1, f5 = T{3}*f4;
    for(Index i = 0; i < nelems; ++i)
    {
        // T z = x[i];
        T z2 = x[i] * x[i];
        T y1 = x[i] * (f3 + f4*z2);
        T y2 = x[i] * (f3 + f5*z2);
        T expy1 = std::exp(y1);
        if(std::isinf(expy1))
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

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *x, const fp32_t *dy, fp32_t *dx)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *x, const fp64_t *dy, fp64_t *dx)
    noexcept;

} // namespace gelutanh_backward
} // namespace kernel
} // namespace nntile