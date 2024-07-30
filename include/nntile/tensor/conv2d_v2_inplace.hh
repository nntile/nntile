/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/conv2d_v2_inplace.hh
 * Tensor wrappers for 2D-Convolution between 2 matrices
 *
 * @version 1.0.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> 2D-Convolution between 2 matrices
template <typename T>
void conv2d_v2_inplace_async(Scalar alpha, const Tensor<T> &src,
        const Tensor<T> &kernel, Scalar beta, const Tensor<T> &dst,
        Index padding_m, Index padding_n);

// Tensor<T> 2D-Convolution between 2 matrices
template <typename T>
void conv2d_v2_inplace(Scalar alpha, const Tensor<T> &src,
        const Tensor<T> &kernel, Scalar beta, const Tensor<T> &dst,
        Index padding_m, Index padding_n);

} // namespace nntile::tensor
