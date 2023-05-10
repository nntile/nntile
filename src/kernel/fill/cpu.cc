/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/fill/cpu.cc
 * Fill operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-24
 * */

#include "nntile/kernel/fill/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace fill
{

template<typename T>
void cpu(Index nelems, T val, T *data)
    noexcept
//! Fill operation on CPU
/*! Sets all elements to the provided value
 * @params[in] nelems: Number of elements in a buffer
 * @param[in] val: Input value
 * @params[out] data: Output buffer
 * */
{
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = val;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t val, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t val, fp64_t *data)
    noexcept;

} // namespace fill
} // namespace kernel
} // namespace nntile

