/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/kernel/sqrt_inplace/cpu.cc
 * Inplace sqrt operation on CPU
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#include "nntile/kernel/sqrt_inplace/cpu.hh"
#include <cmath>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::sqrt_inplace
{

template<typename T>
void cpu(Index nelems, T *data)
    noexcept
//! Inplace sqrt operation on CPU
/*
 * @params[in] nelems: Number of elements in a buffer
 * @params[inout] data_: Buffer to apply sqrt
 * */
{
    using Y = typename T::repr_t;
    for(Index i = 0; i < nelems; ++i)
    {
        data[i] = T{std::sqrt(static_cast<Y>(data[i]))};
    }
}

// Explicit instantiation
template
void cpu<fp64_t>(Index nelems, fp64_t *data)
    noexcept;

template
void cpu<fp32_t>(Index nelems, fp32_t *data)
    noexcept;

template
void cpu<bf16_t>(Index nelems, bf16_t *data)
    noexcept;

} // namespace nntile::kernel::sqrt_inplace
