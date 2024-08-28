/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelu/cpu.cc
 * GeLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/gelu/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::gelu
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept
//! Inplace GeLU operation performed on CPU
/*! Uses very slow std::erfc() function, so consider using approximated version
 * nntile::kernel::cpu::gelutanh(). Does the following per-element operation:
 * GeLU(z) = 0.5 z erfc(-z/sqrt(2))
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply GeLU
 * */
{
    using Y = typename T::repr_t;
    constexpr Y mone{-1.0}, pt5{0.5};
    const Y f1 = mone / std::sqrt(Y{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        Y z(data[i]);
        Y y = std::erfc(f1 * z);
        data[i] = static_cast<T>((pt5 * z) * y);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

template
void cpu<bf16_t>(Index nelems, bf16_t *data)
    noexcept;

} // namespace nntile::kernel::gelu
