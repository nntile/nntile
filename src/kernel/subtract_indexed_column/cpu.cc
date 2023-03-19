/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/subtract_indexed_column/cpu.cc
 * Subtract a value from the indexed column of matrix in a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-18
 * */

#include "nntile/kernel/subtract_indexed_column/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace subtract_indexed_column
{

template<typename T>
void cpu(Index n_row, T val, const Index* class_labels, T *dst)
    noexcept
{
    for (Index i = 0; i < n_row; ++i)
    {
        dst[n_row * class_labels[i] + i] -= val;
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index n_row, fp32_t val, const Index* class_labels, fp32_t *dst)
    noexcept;

template
void cpu<fp64_t>(Index n_row, fp64_t val, const Index* class_labels, fp64_t *dst)
    noexcept;

} // namespace subtract_indexed_column
} // namespace kernel
} // namespace nntile