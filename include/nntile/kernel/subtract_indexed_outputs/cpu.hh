/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_outputs/cpu.hh
 * Subtract a value from certain elements of a matrix on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-06-28
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace subtract_indexed_outputs
{

template<typename T>
void cpu(Index n_labels, Index n_outputs, T val, const Index* labels, T *dst)
    noexcept;

} // namespace subtract_indexed_outputs
} // namespace kernel
} // namespace nntile
