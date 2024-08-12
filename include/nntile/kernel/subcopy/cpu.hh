/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/subcopy/cpu.hh
 * Copy subarray based on contiguous indices
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::subcopy
{

// Complex copying of one multidimensional array into another
template<typename T>
void cpu(Index ndim, const Index *src_start, const Index *src_stride,
        const Index *copy_shape, const T *src, const Index *dst_start,
        const Index *dst_stride, T *dst, int64_t *tmp_index)
    noexcept;

} // namespace nntile::kernel::subcopy
