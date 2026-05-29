/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/log_scalar.hh
 * Log scalar value from Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>
#include <string>

namespace nntile::tensor
{

template<typename T>
void log_scalar_async(const std::string &name, const Tensor<T> &value);

template<typename T>
void log_scalar(const std::string &name, const Tensor<T> &value);

} // namespace nntile::tensor
