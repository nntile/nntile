/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/sqrt_inplace/cpu.cc
 * Inplace sqrt operation on CPU
 *
 * @version 1.0.0
 * */

#include "nntile/kernel/sqrt_inplace/cpu.hh"
#include <cmath>

namespace nntile::kernel::sqrt_inplace
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

} // namespace nntile::kernel::sqrt_inplace

