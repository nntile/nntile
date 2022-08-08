/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/cpu/gelu.cc
 * GeLU operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-04
 * */

#include "nntile/kernel/cpu/gelu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace cpu
{

//! GeLU operation inplace of a buffer on CPU
//
// @params[in] nelems: Number of elements in a buffer
// @params[inout] data: Buffer to apply GeLU
template<typename T>
void gelu(Index nelems, T *data)
    noexcept
{
    constexpr T one = 1, pt5 = 0.5;
    const T sqrt2 = std::sqrt(T{2.0});
    for(Index i = 0; i < nelems; ++i)
    {
        T tmp = pt5*(std::erf(data[i]/sqrt2)) + pt5;
        data[i] *= tmp;
    }
}

// Explicit instantation
template
void gelu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void gelu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

} // namespace cpu
} // namespace kernel
} // namespace nntile

