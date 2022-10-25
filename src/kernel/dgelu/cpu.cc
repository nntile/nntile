/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/dgelu/cpu.cc
 * Derivative of GeLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-24
 * */

#include "nntile/kernel/dgelu/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace dgelu
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept
//! Inplace derivative of GeLU operation performed on CPU
/*! Uses very slow std::erfc() function, so consider using approximated version
 * nntile::kernel::dgelutanh::cpu(). Does the following per-element operation:
 * GeLU'(z) = [0.5 z erfc(-z/sqrt(2))]'
 * GeLU'(z) = 0.5 erfc(-z/sqrt(2)) + [0.5 z (1+erf(z/sqrt(2))']
 * GeLU'(z) = 0.5 erfc(-z/sqrt(2)) + z 1/sqrt(2pi) e^(-z*z/2)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply derivative of GeLU
 * */
{
    constexpr T pi = 3.141592653589793238462643383279502884L,
        one = 1, mone = -1, pt5 = 0.5;
    const T f1 = mone / std::sqrt(T{2.0}), f2 = one / std::sqrt(2*pi);
    for(Index i = 0; i < nelems; ++i)
    {
        T z = data[i];
        T x = std::exp(-pt5 * z * z);
        T y = std::erfc(f1 * z);
        data[i] = z*f2*x + pt5*y;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace dgelu
} // namespace kernel
} // namespace nntile

