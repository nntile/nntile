/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/conv2d_bwd_weight_inplace.hh
 * Backward 2D-Convolution of two tiles in WHCN format to get weight grad
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>

namespace nntile::tile
{

template<typename T>
void conv2d_bwd_weight_inplace_async(Index src1_m, Index src1_n,
        Index src1_channels, Index batch, Index src2_m, Index src2_n,
        Index stride_m, Index stride_n, Index src2_channels, Index offset_m,
        Index offset_n, Scalar alpha, const Tile<T> &src1,
        const Tile<T> &src2, Index dst_m, Index dst_n, Index dilation_m,
        Index dilation_n, Scalar beta, const Tile<T> &dst);

template<typename T>
void conv2d_bwd_weight_inplace(Index src1_m, Index src1_n, Index src1_channels,
        Index batch, Index src2_m, Index src2_n, Index stride_m,
        Index stride_n, Index src2_channels, Index offset_m, Index offset_n,
        Scalar alpha, const Tile<T> &src1, const Tile<T> &src2, Index dst_m,
        Index dst_n, Index dilation_m, Index dilation_n, Scalar beta,
        const Tile<T> &dst);

} // namespace nntile::tile
