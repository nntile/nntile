/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subtract_indexed_column/cpu.hh
 * Subtract value from indexed column of matrix stored in CPU buffer
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace subtract_indexed_column
{

// Compute total sum accumulating from buffers
template<typename T>
void cpu(Index n_row, T val, const Index* class_labels, T *dst)
    noexcept;

} // namespace subtract_indexed_column
} // namespace kernel
} // namespace nntile