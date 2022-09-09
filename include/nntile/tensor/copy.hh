/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/copy.hh
 * Copy operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-08
 * */

#pragma once

#include <nntile/tensor/tensor.hh>
#include <nntile/starpu/copy.hh>

namespace nntile
{
namespace tensor
{

// Asynchronous tensor-wise copy operation
template<typename T>
void copy_async(const Tensor<T> &src, const std::vector<Index> &src_offset,
        const Tensor<T> &dst, const std::vector<Index> &dst_offset);

// Asynchronous tensor-wise copy operation
template<typename T>
void copy_async(const Tensor<T> &src, const Tensor<T> &dst);

// Blocking version of tensor-wise copy operation
template<typename T>
void copy(const Tensor<T> &src, const std::vector<Index> &src_offset,
        const Tensor<T> &dst, const std::vector<Index> &dst_offset);

// Blocking version of tensor-wise copy operation
template<typename T>
void copy(const Tensor<T> &src, const Tensor<T> &dst);

} // namespace tensor
} // namespace nntile

