/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/norm_slice/cpu.hh
 * Euclidian norms of fibers into a slice of a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-02
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace norm_slice
{

// Euclidian norms over fibers along middle axis into a slice of a tensor
template<typename T>
void cpu(Index m, Index n, Index k, T alpha, const T *src, T beta,
        T *dst)
    noexcept;

} // namespace norm_slice
} // namespace kernel
} // namespace nntile

