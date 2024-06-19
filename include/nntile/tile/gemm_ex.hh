/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/gemm_ex.hh
 * GEMM extended operations for Tile<T>
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tile/tile.hh>
#include <nntile/constants.hh>

namespace nntile::tile
{

// Asynchronous tile-wise gemm operation
template<typename T>
void gemm_ex_async(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim, Index batch_ndim);

// Blocking version of tile-wise gemm operation
template<typename T>
void gemm_ex(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim, Index batch_ndim);

} // namespace nntile::tile

