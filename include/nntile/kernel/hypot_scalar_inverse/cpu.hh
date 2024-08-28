/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/hypot_scalar_inverse/cpu.hh
 * Inverse of a hypot operation of a buffer and a scalar
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::hypot_scalar_inverse
{

template<typename T>
void cpu(Index nelems, Scalar eps, Scalar alpha, T* dst)
    noexcept;

} // namespace nntile::kernel::hypot_scalar_inverse
