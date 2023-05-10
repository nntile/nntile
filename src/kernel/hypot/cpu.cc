/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
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
 * @date 2023-04-18
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

//! y := hypot(alpha*x, beta*y)
template<typename T>
void cpu(T alpha, const T *x, T beta, T *y)
    noexcept
{
    constexpr T zero = 0.0;
    if(beta == zero)
    {
        if(alpha == zero)
        {
            y[0] = zero;
        }
        else
        {
            y[0] = std::abs(alpha*x[0]);
        }
    }
    else if(alpha == zero)
    {
        y[0] = std::abs(y[0]*beta);
    }
    else
    {
        y[0] = std::hypot(alpha*x[0], beta*y[0]);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(fp32_t alpha, const fp32_t *x, fp32_t beta, fp32_t *y)
    noexcept;

template
void cpu<fp64_t>(fp64_t alpha, const fp64_t *x, fp64_t beta, fp64_t *y)
    noexcept;

} // namespace hypot
} // namespace kernel
} // namespace nntile

