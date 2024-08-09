/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/add_fiber/cpu.hh
 * Per-element addition of a tensor and a broadcasted fiber on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::add_fiber
{

// Per-element addition of a tensor and a broadcasted fiber on CPU
template<typename T>
void cpu(Index m, Index n, Index k, Index batch, Scalar alpha, const T *src, Scalar beta,
        T *dst)
    noexcept;

} // namespace nntile::kernel::add_fiber
