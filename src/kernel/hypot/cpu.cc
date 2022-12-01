/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/hypot/cpu.cc
 * Hypot of 2 inputs
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-12-01
 * */

#include "nntile/kernel/hypot/cpu.hh"
#include "nntile/base_types.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace hypot
{

//! y:=sqrt(y*y+x*x)
template<typename T>
void cpu(const T *x, T *y)
    noexcept
{
    // Do nothing fancy if x is zero
    if(*x == 0)
    {
        *y = std::abs(*y);
        return;
    }
    T absx = std::abs(*x), absy = std::abs(*y);
    constexpr T one = 1.0;
    if(absx >= absy)
    {
        T tmp = absy / absx;
        *y = absx * std::sqrt(one+tmp*tmp);
    }
    else
    {
        T tmp = absx / absy;
        *y = absy * std::sqrt(one+tmp*tmp);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(const fp32_t *x, fp32_t *y)
    noexcept;

template
void cpu<fp64_t>(const fp64_t *x, fp64_t *y)
    noexcept;

} // namespace hypot
} // namespace kernel
} // namespace nntile

