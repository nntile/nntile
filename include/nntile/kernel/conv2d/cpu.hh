/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d/cpu.hh
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::conv2d
{

template <typename T>
void cpu(Index src_offset_n, Index src_offset_m, Index batch,
         Index out_channels, Index in_channels, Index padding_n,
         Index padding_n2, Index padding_m, Index padding_m2, Index src_n,
         Index src_m, const T *src, Index kernel_n, Index kernel_m,
         const T *kernel, Index dst_n, Index dst_m, T *dst) noexcept;

} // namespace nntile::kernel::conv2d
