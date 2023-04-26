/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/sum_slice/cpu.hh
 * Sums over fibers into a slice of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @author Konstantin Sozykin
 * @date 2023-04-26
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace sum_slice
{

// Sums over fibers along the middle axis into a slice of a tensor
template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta, T *dst)
    noexcept;

} // namespace sum_slice
} // namespace kernel
} // namespace nntile

