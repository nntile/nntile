/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/scal_inplace.hh
 * Inplace scal of Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

//! scal_inplacee tensor
template<typename T>
void scal_inplace_async(Scalar alpha, const Tensor<T> &data);

template<typename T>
void scal_inplace(Scalar alpha, const Tensor<T> &data);

} // namespace nntile::tensor
