/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sgd_step/cpu.hh
 * Fused SGD with momentum step on CPU buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::sgd_step
{

template<typename T>
void cpu(Index num_elems, Scalar momentum, Scalar lr, Scalar weight_decay,
        Scalar dampening, bool nesterov, const T *grad, T *velocity, T *p)
    noexcept;

} // namespace nntile::kernel::sgd_step
