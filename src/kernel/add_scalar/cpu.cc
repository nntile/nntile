/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/add_scalar/cpu.cc
 * Add scalar to element from given buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/kernel/add_scalar/cpu.hh"
#include "nntile/base_types.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace add_scalar
{

template<typename T>
void cpu(T val, Index num_elements, T *y)
    noexcept
{
    // Do nothing fancy if x is zero
    if(val == T(0))
    {
        return;
    }
    for (Index i = 0; i < num_elements; ++i)
        y[i] += val;
}

// Explicit instantiation
template
void cpu<fp32_t>(fp32_t val, Index num_elements, fp32_t *y)
    noexcept;

template
void cpu<fp64_t>(fp64_t val, Index num_elements, fp64_t *y)
    noexcept;

} // namespace add_scalar
} // namespace kernel
} // namespace nntile

