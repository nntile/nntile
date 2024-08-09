/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/conv2d_bwd_input_inplace.hh
 * Backward 2D-Convolution of two tensors in WHCN format to get grad of input
 * Due to Fortran ordering, WHCN of NNTile is equal to NCHF format of PyTorch
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template <typename T>
void conv2d_bwd_input_inplace_async(Scalar alpha, const Tensor<T> &dY,
        const Tensor<T> &kernel, Scalar beta, const Tensor<T> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

template <typename T>
void conv2d_bwd_input_inplace(Scalar alpha, const Tensor<T> &dY,
        const Tensor<T> &kernel, Scalar beta, const Tensor<T> &dX,
        std::array<Index, 2> padding, std::array<Index, 2> stride,
        std::array<Index, 2> dilation);

} // namespace nntile::tensor
