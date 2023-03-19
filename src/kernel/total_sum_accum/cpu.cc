/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/total_sum_accum/cpu.cc
 * Total sum accumulated of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-03-15
 * */

#include "nntile/kernel/total_sum_accum/cpu.hh"
#include <cmath>

namespace nntile
{
namespace kernel
{
namespace total_sum_accum
{

template<typename T>
void cpu(Index n_row, const T* logsumexp, const T* src, const Index* class_labels, T *val)
    noexcept
{
    for (Index i = 0; i < n_row; ++i)
    {
        *val += logsumexp[i] - src[n_row * class_labels[i] + i];
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index n_row, const fp32_t* logsumexp,
                 const fp32_t* src, const Index* class_labels, fp32_t* val)
    noexcept;

template
void cpu<fp64_t>(Index n_row, const fp64_t* logsumexp,
                 const fp64_t* src, const Index* class_labels, fp64_t *val)
    noexcept;

} // namespace total_sum_accum
} // namespace kernel
} // namespace nntile