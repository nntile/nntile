/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/copy_intersection.hh
 * Copy intersection of 2 tensors from one into another
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile::tensor
{

// Asynchronous tensor-wise copy operation
template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset);

// Blocking version of tensor-wise copy operation
template<typename T>
void copy_intersection(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset);

} // namespace nntile::tensor
