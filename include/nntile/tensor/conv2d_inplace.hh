/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/conv2d_inplace.hh
 * Forward 2D-Convolution of two tensors in WHCN format
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Tensor<T> 2D-Convolution between 2 matrices
template <typename T>
void conv2d_inplace_async(Scalar alpha, const Tensor<T> &X,
        const Tensor<T> &C, Scalar beta, const Tensor<T> &Y,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

// Tensor<T> 2D-Convolution between 2 matrices
template <typename T>
void conv2d_inplace(Scalar alpha, const Tensor<T> &X,
        const Tensor<T> &C, Scalar beta, const Tensor<T> &Y,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

} // namespace nntile::tensor
