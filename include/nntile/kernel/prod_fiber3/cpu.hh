/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/prod_fiber3/cpu.hh
 * Per-element multiplication of a tensor by a broadcasted fiber on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::prod_fiber3
{

// Per-element product of a tensor and a broadcasted fiber on CPU
template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha, const T *src1, const T *src2,
        T *dst)
    noexcept;

} // namespace nntile::kernel::prod_fiber3
