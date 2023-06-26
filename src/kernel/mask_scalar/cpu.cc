/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/mask_scalar/cpu.cc
 * Mask operation with scalar on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-22
 * */

#include "nntile/kernel/mask_scalar/cpu.hh"

namespace nntile
{
namespace kernel
{
namespace mask_scalar
{

template<typename T>
void cpu(Index nelems, bool_t* mask, T val, T *data)
    noexcept
//! Mask operation with sclar on CPU
/*! Sets all elements to the provided value if the mask value is 0
 * @params[in] nelems: Number of elements in a buffer
 * @params[in] mask: buffer with mask values
 * @params[in] val: value to set if mask element is 0
 * @params[in,out] data: Input data and result of mask transformation
 * */
{
    for(Index i = 0; i < nelems; ++i)
    {
        if (!mask[i])
        {
            data[i] = val;
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, bool_t* mask, fp32_t val, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, bool_t* mask, fp64_t val, fp64_t *data)
    noexcept;

} // namespace mask_scalar
} // namespace kernel
} // namespace nntile

