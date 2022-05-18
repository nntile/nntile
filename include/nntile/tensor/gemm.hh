/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/gemm.hh
 * GEMM operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/constants.hh>

namespace nntile
{

//! Asynchronous tensor-wise gemm operation
//
// @param[in] alpha: Alpha multiplier
// @param[in] transA: Transposition flag for the tensor A
// @param[in] A: Input tensor A
// @param[in] transB: Transposition flag for the tensor B
// @param[in] B: Input tensor B
// @param[in] beta: Beta multiplier
// @param[inout] C: Output tensor C
// @param[in] ndim: Number of dimensions used in gemm contraction
template<typename T>
void gemm_work(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim=1);

extern template
void gemm_work(fp32_t alpha, const TransOp &transA, const Tensor<fp32_t> &A,
        const TransOp &transB, const Tensor<fp32_t> &B, fp32_t beta,
        const Tensor<fp32_t> &C, Index ndim=1);

extern template
void gemm_work(fp64_t alpha, const TransOp &transA, const Tensor<fp64_t> &A,
        const TransOp &transB, const Tensor<fp64_t> &B, fp64_t beta,
        const Tensor<fp64_t> &C, Index ndim=1);

void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim=1);

template<typename T>
void gemm_async(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim=1)
{
    // Check inputs (throw exception in case of an error)
    gemm_check(transA, A, transB, B, C, ndim);
    // Launch all gemms
    gemm_work<T>(alpha, transA, A, transB, B, beta, C, ndim);
}

//! Blocking version of tensor-wise gemm operation
//
// @param[in] alpha: Alpha multiplier
// @param[in] transA: Transposition flag for the tensor A
// @param[in] A: Input tensor A
// @param[in] transB: Transposition flag for the tensor B
// @param[in] B: Input tensor B
// @param[in] beta: Beta multiplier
// @param[inout] C: Output tensor C
// @param[in] ndim: Number of dimensions used in gemm contraction
template<typename T>
void gemm(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim=1)
{
    gemm_async<T>(alpha, transA, A, transB, B, beta, C, ndim);
    starpu_task_wait_for_all();
}

} // namespace nntile

