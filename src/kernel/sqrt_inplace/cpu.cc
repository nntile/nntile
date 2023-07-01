/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sqrt_inplace/cpu.cc
 * Inplace sqrt operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-07-01
 * */

#include "nntile/kernel/sqrt_inplace/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace sqrt_inplace
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept
//! Inplace sqrt operation on CPU
/*
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply sqrt
 * */
{
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = std::sqrt(data[i]);
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace sqrt_inplace
} // namespace kernel
} // namespace nntile

