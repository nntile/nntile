/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/lars_tiled_step/cpu.hh
 * Fused LarsTiled step on CPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::lars_tiled_step
{

template<typename T>
void cpu(Index num_iter, Index num_elems, Index num_steps,
    Scalar gamma_0, Scalar momentum, Scalar weight_decay, Scalar lars_coefficient,
    const T *grad, T *momentum_buffer, T *p)
    noexcept;

} // namespace nntile::kernel::lars_tiled_step
