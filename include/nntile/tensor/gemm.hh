/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
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
 * @date 2023-05-04
 * */

#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/constants.hh>

namespace nntile
{
namespace tensor
{

void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim, Index batch_ndim);

template<typename T, typename T_scal>
void gemm_async(T_scal alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T_scal beta,
        const Tensor<T> &C,
        Index ndim, Index batch_ndim);

template<typename T, typename T_scal>
void gemm(T_scal alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T_scal beta,
        const Tensor<T> &C,
        Index ndim, Index batch_ndim);

} // namespace tensor
} // namespace nntile

