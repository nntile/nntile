/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/gemm.hh
 * GEMM operation for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-08-08
 * */

#pragma once

#include <nntile/tile/tile.hh>
#include <nntile/constants.hh>

namespace nntile
{

void gemm_check(const TransOp &transA, const TileTraits &A,
        const TransOp &transB, const TileTraits &B, const TileTraits &C,
        Index ndim=1);

//! Asynchronous tile-wise gemm operation
//
// @param[in] alpha: Alpha multiplier
// @param[in] transA: Transposition flag for the tile A
// @param[in] A: Input tile A
// @param[in] transB: Transposition flag for the tile B
// @param[in] B: Input tile B
// @param[in] beta: Beta multiplier
// @param[inout] C: Output tile C
// @param[in] ndim: Number of dimensions used in gemm contraction
template<typename T>
void gemm_work(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim=1);

template<typename T>
void gemm_async(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim=1)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim);
    // Launch codelet
    gemm_work<T>(alpha, transA, A, transB, B, beta, C, ndim);
}

//! Blocking version of tile-wise gemm operation
//
// @param[in] alpha: Alpha multiplier
// @param[in] transA: Transposition flag for the tile A
// @param[in] A: Input tile A
// @param[in] transB: Transposition flag for the tile B
// @param[in] B: Input tile B
// @param[in] beta: Beta multiplier
// @param[inout] C: Output tile C
// @param[in] ndim: Number of dimensions used in gemm contraction
template<typename T>
void gemm(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim=1)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
}

} // namespace nntile

