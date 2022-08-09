/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/cpu/randn.hh
 * Randn operation on a buffer on CPU
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-09
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile
{
namespace kernel
{
namespace cpu
{

template<typename T>
void randn(Index ndim, Index nelems, unsigned long long seed,
        T mean, T stddev, const Index *start, const Index *shape,
        const Index *underlying_shape, T *data, const Index *stride,
        Index *tmp_index)
    noexcept;

} // namespace cpu
} // namespace kernel
} // namespace nntile
