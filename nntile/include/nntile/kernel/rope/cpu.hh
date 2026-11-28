/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/rope/cpu.hh
 * Rotary positional embedding
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::rope
{

template<typename T>
void cpu(Index m, Index n, const T *sin, const T *cos, const T *src, T *dst)
    noexcept;

} // namespace nntile::kernel::rope
