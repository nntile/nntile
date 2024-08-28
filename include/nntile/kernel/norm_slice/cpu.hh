/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/norm_slice/cpu.hh
 * Euclidean norms of fibers into a slice of a buffer on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::norm_slice
{

// Euclidean norms over fibers along middle axis into a slice of a tensor
template<typename T>
void cpu(Index m, Index n, Index k, Scalar alpha, const T *src, Scalar beta,
        T *dst)
    noexcept;

} // namespace nntile::kernel::norm_slice
