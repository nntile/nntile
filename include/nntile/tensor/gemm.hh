/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/gemm.hh
 * GEMM operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/constants.hh>

namespace nntile::tensor
{

void gemm_check(const TransOp &transA, const TensorTraits &A,
        const TransOp &transB, const TensorTraits &B, const TensorTraits &C,
        Index ndim, Index batch_ndim);

template<typename T>
void gemm_async(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux=0);

template<typename T>
void gemm(Scalar alpha, const TransOp &transA, const Tensor<T> &A,
        const TransOp &transB, const Tensor<T> &B, Scalar beta,
        const Tensor<T> &C, Index ndim, Index batch_ndim, int redux=0);

} // namespace nntile::tensor
