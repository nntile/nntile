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
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tile/tile.hh>
#include <nntile/constants.hh>

namespace nntile
{

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

extern template
void gemm_work(fp32_t alpha, const TransOp &transA, const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, fp32_t beta,
        const Tile<fp32_t> &C, Index ndim=1);

extern template
void gemm_work(fp64_t alpha, const TransOp &transA, const Tile<fp64_t> &A,
        const TransOp &transB, const Tile<fp64_t> &B, fp64_t beta,
        const Tile<fp64_t> &C, Index ndim=1);

void gemm_check(const TransOp &transA, const TileTraits &A,
        const TransOp &transB, const TileTraits &B, const TileTraits &C,
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

extern starpu_perfmodel perfmodel_gemm_fp32, perfmodel_gemm_fp64;
extern StarpuCodelet codelet_gemm_fp32, codelet_gemm_fp64;
void gemm_restrict_where(uint32_t where);
void gemm_restore_where();

template<typename T>
constexpr StarpuCodelet *gemm_get_codelet()
{
    throw std::runtime_error("Non-supported type");
    return nullptr;
}

template<>
constexpr StarpuCodelet *gemm_get_codelet<fp32_t>()
{
    return &codelet_gemm_fp32;
}

template<>
constexpr StarpuCodelet *gemm_get_codelet<fp64_t>()
{
    return &codelet_gemm_fp64;
}

} // namespace nntile

