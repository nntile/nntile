/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/dgelu/cpu.cc
 * Derivative of GeLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/dgelu/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::dgelu
{

template<typename T>
void cpu(Index nelems, T *data_)
    noexcept
//! Inplace derivative of GeLU operation performed on CPU
/*! Uses very slow std::erfc() function, so consider using approximated version
 * nntile::kernel::dgelutanh::cpu(). Does the following per-element operation:
 * GeLU'(z) = [0.5 z erfc(-z/sqrt(2))]'
 * GeLU'(z) = 0.5 erfc(-z/sqrt(2)) + [0.5 z (1+erf(z/sqrt(2))']
 * GeLU'(z) = 0.5 erfc(-z/sqrt(2)) + z 1/sqrt(2pi) e^(-z*z/2)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply derivative of GeLU
 * */
{
    using Y = typename CPUComputeType<T>::value;
    auto data = reinterpret_cast<Y *>(data_);
    constexpr Y pi{3.141592653589793238462643383279502884L},
        one{1.0}, mone{-1.0}, pt5{0.5};
    const Y f1 = mone / std::sqrt(Y{2.0}), f2 = one / std::sqrt(2*pi);
    for(Index i = 0; i < nelems; ++i)
    {
        Y z = data[i];
        Y x = std::exp(-pt5 * z * z);
        Y y = std::erfc(f1 * z);
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

} // namespace nntile::kernel::dgelu
