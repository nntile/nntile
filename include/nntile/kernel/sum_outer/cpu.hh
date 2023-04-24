/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_outer/cpu.hh
 * Sum of slices of a buffer on CPU (outer version)
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-04-24
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace sum_outer
{

// Compute sum along outer axis
template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *sum_dst)
    noexcept;

} // namespace sum_outer
} // namespace kernel
} // namespace nntile

