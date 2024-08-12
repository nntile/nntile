/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/randn/cpu.hh
 * Randn operation on a buffer on CPU
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::randn
{

template<typename T>
void cpu(Index ndim, Index nelems, unsigned long long seed,
        Scalar mean, Scalar stddev, const Index *start, const Index *shape,
        const Index *underlying_shape, T *data, const Index *stride,
        int64_t *tmp_index)
    noexcept;

template<typename T>
void cpu_ndim0(unsigned long long seed, Scalar mean, Scalar stddev, T *data)
    noexcept;

} // namespace nntile::kernel::randn
