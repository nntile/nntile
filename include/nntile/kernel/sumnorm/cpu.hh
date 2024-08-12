/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sumnorm/cpu.hh
 * Sum and Euclidean norm of a buffer on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::sumnorm
{

// Compute sum and Euclidean norm along middle axis
template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *sumnorm)
    noexcept;

} // namespace nntile::kernel::sumnorm
