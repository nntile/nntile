/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelutanh_inplace/cpu.cc
 * Approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelutanh_inplace/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::gelutanh_inplace
{

template<typename T>
void cpu(Index nelems, T *data_)
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
 * @params[inout] data_: Buffer to apply GeLU
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto data = reinterpret_cast<Y *>(data_);
    // Constants
    constexpr Y pi{3.141592653589793238462643383279502884L},
        one{1.0}, pt5{0.5}, f1{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    static const Y sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2.0}*f2, f4 = f3*f1;
    for(Index i = 0; i < nelems; ++i)
    {
        Y z = data[i];
        Y y = z * (f3 + f4*z*z);
        data[i] = z / (one+std::exp(y));
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::gelutanh_inplace
