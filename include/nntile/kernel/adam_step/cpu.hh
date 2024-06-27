/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/adam_step/cpu.hh
 * Fused Adam step on CPU buffers
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::adam_step
{

template<typename T>
void cpu(Index num_iter, Index num_elems, scal_t beta_1, scal_t beta_2,
        scal_t eps, scal_t lr, scal_t weight_decay, const T *grad,
        T *first_moment, T *second_moment, T *p)
    noexcept;

} // namespace nntile::kernel::adam_step
