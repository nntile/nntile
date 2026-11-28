/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file include/nntile/kernel/conv2d_bwd_weight_inplace/cpu.hh
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of weight
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 2.0.0
 * */
/*!
 * INTENTIONALLY DISABLED in the NNTile 2.0 transformer-only scope.
 * Kept in-tree for future re-enablement. No product path calls this code;
 * its tests are not registered while the family is in
 * NNTILE_DISABLED_KERNELS.
 */
#pragma once

#include <nntile/base_types.hh>

namespace nntile::kernel::conv2d_bwd_weight_inplace
{

template<typename T>
void cpu(Index src1_m, Index src1_n, Index src1_channels, Index batch,
        Index src2_m, Index src2_n, Index stride_m, Index stride_n,
        Index src2_channels, Index offset_m, Index offset_n, Scalar alpha,
        const T *src1, const T *src2, Index dst_m, Index dst_n,
        Index dilation_m, Index dilation_n, Scalar beta, T *dst)
    noexcept;

} // namespace nntile::kernel::conv2d_bwd_weight_inplace
