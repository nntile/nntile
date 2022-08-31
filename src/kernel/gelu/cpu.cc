/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/gelu/cpu.cc
 * GeLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-31
 * */

#include "nntile/kernel/gelu/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace gelu
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
    constexpr T mone = -1, pt5 = 0.5;
    const T f1 = mone / std::sqrt(T{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        T z = data[i];
        T y = std::erfc(f1 * z);
        data[i] = pt5 * z * y;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace gelu
} // namespace kernel
} // namespace nntile

