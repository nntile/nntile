/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_outputs/cpu.hh
 * Subtract a value from certain elements of a matrix on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::subtract_indexed_outputs
{

template<typename T>
void cpu(Index n_labels, Index n_outputs, Scalar val, const int64_t* labels, T *dst)
    noexcept;

} // namespace nntile::kernel::subtract_indexed_outputs
