/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/conv2d_v2_inplace/cpu.hh
 * 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::conv2d_v2_inplace
{

template <typename T>
void cpu(Index src_m, Index src_n, Index in_channels, Index batch,
        Index offset_m, Index offset_n, Scalar alpha, const T *src,
        Index kernel_m, Index kernel_n, Index out_channels, const T *kernel,
        Index dst_m, Index dst_n, Scalar beta, T *dst) noexcept;

} // namespace nntile::kernel::conv2d_v2_inplace
