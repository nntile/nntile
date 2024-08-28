/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/dgelutanh/cpu.cc
 * Derivative of approximate GeLU operation on CPU based on tanh function
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/dgelutanh/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::dgelutanh
{

template<typename T>
void cpu(Index nelems, T *data_)
    noexcept
//! Derivative of approximate GeLU operation on CPU
/*! Applies the following derivative of approximation of the GeLU function:
 * GeLU(z) \approx AGeLU(z)
 * f(z) = -2 sqrt(2/pi) z (1+0.044715z^2)
 * AGeLU(z) = z / (1+exp(f(z))
 * AGeLU'(z) = 1/(1+exp(f(z)) - (zf'(z)exp(f(z)))/(1+exp(f(z)))^2
 * AGeLU'(z) = (1-(zf'(z)-1)exp(f(z))) / (1+exp(f(z)))^2
 * zf'(z) = -2 sqrt(2/pi) z (1+3*0.044715z^2)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply derivative of approximate GeLU
 * */
{
    // Constants
    using Y = typename CPUComputeType<T>::value;
    auto data = reinterpret_cast<Y *>(data_);
    constexpr Y pi{3.141592653589793238462643383279502884L},
        zero{0.0}, one{1.0}, pt5{0.5}, f1 = Y{0.044715};
    // Square root is not constexpr by standard, proceed with a static const
    static const Y sqrt_pi = std::sqrt(pi), sqrt_2 = std::sqrt(Y{2.0}),
        f2 = sqrt_2/sqrt_pi, f3 = -Y{2.0}*f2, f4 = f3*f1, f5 = Y{3.0}*f4;
    for(Index i = 0; i < nelems; ++i)
    {
        Y z = data[i];
        Y z2 = z * z;
        Y y1 = z * (f3 + f4*z2);
        Y y2 = z * (f3 + f5*z2);
        Y expy1 = std::exp(y1);
        if(std::isinf(expy1))
        {
            data[i] = zero;
        }
        else
        {
            Y inv_expy1p1 = one / (expy1 + one);
            data[i] = (one-y2*(one-inv_expy1p1)) * inv_expy1p1;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace nntile::kernel::dgelutanh
