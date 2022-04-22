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
 * @date 2022-04-22
 * */

#pragma once

#include <nntile/tensor/tensor.hh>

namespace nntile
{

//! Asynchronous tensor-wise copy operation
//
// @param[in] src: Source tensor
// @param[in] src_offset: Initial offset of the source tensor
// @param[inout] dst: Destination tensor
// @param[in] dst_offset: Initial offset of the destination tensor
//
// This operation finds an intersection of the source and the target tensors
// and copies only the data within the found intersection. No elements of the
// destination tensor outside the intersection mask are updated.
template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset,
        const Tensor<T> &dst, const std::vector<Index> &dst_offset);

extern template
void copy_intersection_async(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset);

extern template
void copy_intersection_async(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset);

//! Asynchronous tensor-wise copy operation
//
// @param[in] src: Source tensor
// @param[inout] dst: Destination tensor
//
// This operation finds an intersection of the source and the target tensors
// and copies only the data within the found intersection. No elements of the
// destination tensor outside the intersection mask are updated. Both the
// source and the target tensors assumed to have the same offset.
template<typename T>
void copy_intersection_async(const Tensor<T> &src, const Tensor<T> &dst)
{
    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
            std::vector<Index>(dst.ndim));
}

//! Blocking version of tensor-wise copy operation
//
// @param[in] src: Source tensor
// @param[in] src_offset: Initial offset of the source tensor
// @param[inout] dst: Destination tensor
// @param[in] dst_offset: Initial offset of the destination tensor
//
// This operation finds an intersection of the source and the target tensors
// and copies only the data within the found intersection. No elements of the
// destination tensor outside the intersection mask are updated.
template<typename T>
void copy_intersection(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset)
{
    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
    starpu_task_wait_for_all();
}

//! Blocking version of tensor-wise copy operation
//
// @param[in] src: Source tensor
// @param[inout] dst: Destination tensor
//
// This operation finds an intersection of the source and the target tensors
// and copies only the data within the found intersection. No elements of the
// destination tensor outside the intersection mask are updated. Both the
// source and the target tensors assumed to have the same offset.
template<typename T>
void copy_intersection(const Tensor<T> &src, const Tensor<T> &dst)
{
    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
            std::vector<Index>(dst.ndim));
    starpu_task_wait_for_all();
}

} // namespace nntile

