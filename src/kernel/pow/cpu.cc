/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/pow/cpu.cc
 * Power operation on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-07-07
 * */

#include "nntile/kernel/pow/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace pow
{

template<typename T>
void cpu(Index nelems, T alpha, T exp, T *data)
    noexcept
//! Inplace power operation on CPU
/*! Does the following per-element operation:
 * pow(z) = alpha * z^exp
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply power function
 * */
{
    for(Index i = 0; i < nelems; ++i)
    {
        T z = data[i];
        if(exp == -1)
        {
            data[i] = alpha / z;
        }
        else
        {
            data[i] = alpha * std::pow(z, exp);
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t alpha, fp32_t exp, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t alpha, fp64_t exp, fp64_t *data)
    noexcept;

} // namespace pow
} // namespace kernel
} // namespace nntile

