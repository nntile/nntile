/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/maxsumexp/cpu.hh
 * Max and sum of exponents of a buffer on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::maxsumexp
{

// Compute max and sums of exponents along middle axis
template<typename T>
void cpu(Index m, Index n, Index k, const T *src, T *maxsumexp)
    noexcept;

} // namespace nntile::kernel::maxsumexp
