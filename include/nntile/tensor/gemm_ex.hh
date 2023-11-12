/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/gemm_ex.hh
 * GEMM extended operations for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-11-12
 * */

#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/constants.hh>

namespace nntile
{
namespace tensor
{

template<typename T>
void gemm_ex_async(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim, Index batch_ndim, int redux=0);

template<typename T>
void gemm_ex(T alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, T beta, const Tensor<T> &C,
        Index ndim, Index batch_ndim, int redux=0);

} // namespace tensor
} // namespace nntile

