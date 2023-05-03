/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/gemm_ex.cc
 * GEMM extended operations for Tile<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-05-03
 * */

#include "nntile/tile/gemm_ex.hh"
#include "nntile/tile/gemm.hh"
#include "nntile/starpu/gemm_ex.hh"

namespace nntile
{
namespace tile
{

template<typename T>
void gemm_ex_async(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim, Index batch_ndim)
//! Asynchronous tile-wise gemm operation
/*! @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tile A
 * @param[in] A: Input tile A
 * @param[in] transB: Transposition flag for the tile B
 * @param[in] B: Input tile B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tile C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * */
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim, batch_ndim);
    // Reference tensors as matrices
    Index m = C.matrix_shape[A.ndim-batch_ndim-ndim][0];
    Index batch = C.matrix_shape[C.ndim-batch_ndim][1];
    Index n = C.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
    Index k;
    switch(transA.value)
    {
        case TransOp::NoTrans:
            k = A.matrix_shape[A.ndim-batch_ndim-ndim][1] / batch;
            break;
        // This parameter was already checked in gemm_check_opA_opB
        //case TransOp::Trans:
        default:
            k = A.matrix_shape[ndim][0];
            break;
    }
    // Insert task
    starpu::gemm_ex::submit<T>(transA, transB, m, n, k, batch, alpha, A, B,
            beta, C);
}

template<typename T>
void gemm_ex(T alpha, const TransOp &transA, const Tile<T> &A,
        const TransOp &transB, const Tile<T> &B, T beta, const Tile<T> &C,
        Index ndim, Index batch_ndim)
//! Blocking version of tile-wise gemm operation
/*! @param[in] alpha: Alpha multiplier
 * @param[in] transA: Transposition flag for the tile A
 * @param[in] A: Input tile A
 * @param[in] transB: Transposition flag for the tile B
 * @param[in] B: Input tile B
 * @param[in] beta: Beta multiplier
 * @param[inout] C: Output tile C
 * @param[in] ndim: Number of dimensions used in gemm contraction
 * @param[in] batch_ndim: Number of last dimensions used for batching of gemms
 * */
{
    gemm_ex_async<T>(alpha, transA, A, transB, B, beta, C, ndim, batch_ndim);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void gemm_ex_async<fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, fp32_t beta,
        const Tile<fp32_t> &C, Index ndim, Index batch_ndim);

// Explicit instantiation
template
void gemm_ex<fp32_t>(fp32_t alpha, const TransOp &transA,
        const Tile<fp32_t> &A,
        const TransOp &transB, const Tile<fp32_t> &B, fp32_t beta,
        const Tile<fp32_t> &C, Index ndim, Index batch_ndim);

} // namespace tile
} // namespace nntile

