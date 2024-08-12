/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh/cpu.cc
 * Approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::gelutanh
{

template<typename T>
void cpu(Index nelems, const T *src, T *dst)
    noexcept
//! Approximate GeLU operation on CPU
/*! Applies the following approximation of the GeLU function:
 * GeLU(z) \approx AGeLU(z)
 * AGeLU(z) \approx 0.5z(1+tanh(sqrt(2/pi)(z+0.044715z^3))),
 * which is actually implemented as
 * f(z) = -2 sqrt(2/pi) z (1+0.044715z^2)
 * AGeLU(z) = z / (1+exp(f(z))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] src_: Input buffer to apply GeLU
 * @params[out] dst_: Output buffer to apply GeLU
 * */
{
    using Y = typename T::repr_t;
    // Constants
    constexpr Y pi{3.141592653589793238462643383279502884L},
        one{1.0}, pt5{0.5}, f1{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    static const Y sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2.0}*f2, f4 = f3*f1;
    for(Index i = 0; i < nelems; ++i)
    {
        Y z = Y{src[i]};
        Y y1 = f4 * z * z;
        Y y2 = f3 + y1;
        Y c = y1 - (y2-f3);
        y2 *= z;
        c *= z;
        Y y3 = one + std::exp(c)*std::exp(y2);
        dst[i] = T{z / y3};
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, const fp32_t *src, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index nelems, const fp64_t *src, fp64_t *dst)
    noexcept;

template
void cpu<bf16_t>(Index nelems, const bf16_t *src, bf16_t *dst)
    noexcept;

} // namespace nntile::kernel::gelutanh
