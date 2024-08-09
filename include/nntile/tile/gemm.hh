/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/gemm.hh
 * GEMM operation for Tile<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/tile.hh>
#include <nntile/constants.hh>

namespace nntile::tile
{

// Check if tensors match gemm
void gemm_check(const TransOp &transA, const TileTraits &A,
        const TransOp &transB, const TileTraits &B, const TileTraits &C,
        Index ndim, Index batch_ndim);

// Asynchronous tile-wise gemm operation
template<typename T>
void gemm_async(Scalar alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, Scalar beta, const Tile<T> &C,
        Index ndim, Index batch_ndim);

// Blocking version of tile-wise gemm operation
template<typename T>
void gemm(Scalar alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, Scalar beta, const Tile<T> &C,
        Index ndim, Index batch_ndim);

} // namespace nntile::tile
