/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/silu_inplace/cpu.cc
 * Inplace SiLU operation on CPU
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/silu_inplace/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::silu_inplace
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept
//! Inplace SiLU operation on CPU
/*! Does the following per-element operation:
 * data[i] = data[i] * sigmoid(data[i])
 *
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data: Buffer to apply SiLU
 * */
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < nelems; ++i)
    {
        Y data_val{data[i]};
        data[i] = static_cast<T>(data_val / (Y{1.} + std::exp(-data_val)));
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

template
void cpu<bf16_t>(Index nelems, bf16_t *data)
    noexcept;

template
void cpu<fp16_t>(Index nelems, fp16_t *data)
    noexcept;

} // namespace nntile::kernel::silu_inplace