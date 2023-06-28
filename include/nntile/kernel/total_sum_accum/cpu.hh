/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/total_sum_accum/cpu.hh
 * Total sum accumulated of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @author Aleksandr Mikhalev
 * @date 2023-06-28
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace total_sum_accum
{

// Compute total sum accumulating from buffers
template<typename T>
void cpu(Index n_labels, Index n_outputs, const T* logsumexp, const T* src,
        const Index* labels, T *val)
    noexcept;

} // namespace total_sum_accum
} // namespace kernel
} // namespace nntile
