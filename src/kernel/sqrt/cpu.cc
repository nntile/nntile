/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/relu/cpu.cc
 * ReLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/kernel/sqrt/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace sqrt
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

} // namespace sqrt
} // namespace kernel
} // namespace nntile

