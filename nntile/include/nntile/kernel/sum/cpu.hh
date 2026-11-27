/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum/cpu.hh
 * Sum all elements of a buffer on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::sum
{

// Sum all elements of a tensor into a scalar
template<typename T>
void cpu(Index nelems, Scalar alpha, const T *src, Scalar beta, T *dst)
    noexcept;

} // namespace nntile::kernel::sum
