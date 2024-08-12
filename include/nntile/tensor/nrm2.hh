/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/nrm2.hh
 * Euclidean norm of Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

//! Compute Euclidean norm
template<typename T>
void nrm2_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        const Tensor<T> &tmp);

template<typename T>
void nrm2(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst,
        const Tensor<T> &tmp);

} // namespace nntile::tensor
