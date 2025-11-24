/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/norm.hh
 * Euclidean norm of all elements in a Tensor<T>
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/norm_traits.hh>
#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

template<typename T>
using norm_dst_tensor_t = Tensor<norm_value_t<T>>;

// Tensor-wise norm
template<typename T>
void norm_async(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const norm_dst_tensor_t<T> &dst);

// Tensor-wise norm
template<typename T>
void norm(Scalar alpha, const Tensor<T> &src, Scalar beta,
        const norm_dst_tensor_t<T> &dst);

} // namespace nntile::tensor
