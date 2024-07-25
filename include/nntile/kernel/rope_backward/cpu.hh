/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/rope_backward/cpu.hh
 * Rotary positional embedding
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::rope_backward
{

template<typename T>
void cpu(Index m, Index n, const T *sin, const T *cos, const T *dy, T *dx)
    noexcept;

} // namespace nntile::kernel::rope_backward
