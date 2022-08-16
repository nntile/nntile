/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/relu.cc
 * ReLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-16
 * */

#include "nntile/kernel/cpu/relu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace cpu
{

template<typename T>
void relu(Index nelems, T *data)
    noexcept
//! Inplace ReLU operation on CPU
/*! Does the following per-element operation:
 * ReLU(z) = max(z, 0)
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply ReLU
 * */
{
    constexpr T zero = 0;;
    for(Index i = 0; i < nelems; ++i)
    {
        T z = data[i];
        data[i] = std::max(z, zero);
    }
}

// Explicit instantiation
template
void relu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void relu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace cpu
} // namespace kernel
} // namespace nntile

