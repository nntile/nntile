/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum/cpu.hh
 * Total sum accumulated of a buffer on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::total_sum_accum
{

// Compute total sum accumulating from buffers
template<typename T>
void cpu(Scalar alpha, Index n_labels, Index n_outputs, const T* logsumexp,
        const T* src, const int64_t* labels, float *val)
    noexcept;

} // namespace nntile::kernel::total_sum_accum
