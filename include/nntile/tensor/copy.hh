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

////! Asynchronous copy from tile to tensoroperation
////
//// @param[in] src: Source tile
//// @param[in] src_offset: Initial offset of the source tile
//// @param[inout] dst: Destination tensor
//// @param[in] dst_offset: Initial offset of the destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated.
//template<typename T>
//void copy_intersection_work(const Tile<T> &src,
//        const std::vector<Index> &src_offset, const Tensor<T> &dst,
//        const std::vector<Index> &dst_offset,
//        const StarpuVariableHandle &scratch);
//
//extern template
//void copy_intersection_work(const Tile<fp32_t> &src,
//        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
//        const std::vector<Index> &dst_offset,
//        const StarpuVariableHandle &scratch);
//
//extern template
//void copy_intersection_work(const Tile<fp64_t> &src,
//        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
//        const std::vector<Index> &dst_offset,
//        const StarpuVariableHandle &scratch);
//
//template<typename T>
//void copy_intersection_async(const Tile<T> &src,
//        const std::vector<Index> &src_offset,
//        const Tensor<T> &dst, const std::vector<Index> &dst_offset)
//{
//    // Check dimensions
//    if(src.ndim != src_offset.size())
//    {
//        throw std::runtime_error("src.ndim != src_offset.size()");
//    }
//    if(src.ndim != dst.ndim)
//    {
//        throw std::runtime_error("src.ndim != dst.ndim");
//    }
//    if(dst.ndim != dst_offset.size())
//    {
//        throw std::runtime_error("dst.ndim != dst_offset.size()");
//    }
//    // Temporary buffer for indexing, that is allocated per-worker when needed
//    StarpuVariableHandle scratch(2 * src.ndim * sizeof(Index));
//    // Launch codelet
//    copy_intersection_work(src, src_offset, dst, dst_offset, scratch);
//}
//
////! Asynchronous tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[inout] dst: Destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated. Both the
//// source and the target tensors assumed to have the same offset.
//template<typename T>
//void copy_intersection_async(const Tile<T> &src, const Tensor<T> &dst)
//{
//    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
//            std::vector<Index>(dst.ndim));
//}
//
////! Blocking version of tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[in] src_offset: Initial offset of the source tensor
//// @param[inout] dst: Destination tensor
//// @param[in] dst_offset: Initial offset of the destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated.
//template<typename T>
//void copy_intersection(const Tile<T> &src,
//        const std::vector<Index> &src_offset, const Tensor<T> &dst,
//        const std::vector<Index> &dst_offset)
//{
//    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
//    starpu_task_wait_for_all();
//}
//
////! Blocking version of tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[inout] dst: Destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated. Both the
//// source and the target tensors assumed to have the same offset.
//template<typename T>
//void copy_intersection(const Tile<T> &src, const Tensor<T> &dst)
//{
//    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
//            std::vector<Index>(dst.ndim));
//    starpu_task_wait_for_all();
//}
//
////! Asynchronous tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[in] src_offset: Initial offset of the source tensor
//// @param[inout] dst: Destination tensor
//// @param[in] dst_offset: Initial offset of the destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated.
//template<typename T>
//void copy_intersection_work(const Tensor<T> &src,
//        const std::vector<Index> &src_offset, const Tile<T> &dst,
//        const std::vector<Index> &dst_offset,
//        const StarpuVariableHandle &scratch);
//
//extern template
//void copy_intersection_work(const Tensor<fp32_t> &src,
//        const std::vector<Index> &src_offset, const Tile<fp32_t> &dst,
//        const std::vector<Index> &dst_offset,
//        const StarpuVariableHandle &scratch);
//
//extern template
//void copy_intersection_work(const Tensor<fp64_t> &src,
//        const std::vector<Index> &src_offset, const Tile<fp64_t> &dst,
//        const std::vector<Index> &dst_offset,
//        const StarpuVariableHandle &scratch);
//
//template<typename T>
//void copy_intersection_async(const Tensor<T> &src,
//        const std::vector<Index> &src_offset,
//        const Tile<T> &dst, const std::vector<Index> &dst_offset)
//{
//    // Check dimensions
//    if(src.ndim != src_offset.size())
//    {
//        throw std::runtime_error("src.ndim != src_offset.size()");
//    }
//    if(src.ndim != dst.ndim)
//    {
//        throw std::runtime_error("src.ndim != dst.ndim");
//    }
//    if(dst.ndim != dst_offset.size())
//    {
//        throw std::runtime_error("dst.ndim != dst_offset.size()");
//    }
//    // Temporary buffer for indexing, that is allocated per-worker when needed
//    StarpuVariableHandle scratch(2 * src.ndim * sizeof(Index));
//    // Launch codelets for non-zero ndim
//    copy_intersection_work(src, src_offset, dst, dst_offset, scratch);
//}
//
////! Asynchronous tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[inout] dst: Destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated. Both the
//// source and the target tensors assumed to have the same offset.
//template<typename T>
//void copy_intersection_async(const Tensor<T> &src, const Tile<T> &dst)
//{
//    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
//            std::vector<Index>(dst.ndim));
//}
//
////! Blocking version of tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[in] src_offset: Initial offset of the source tensor
//// @param[inout] dst: Destination tensor
//// @param[in] dst_offset: Initial offset of the destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated.
//template<typename T>
//void copy_intersection(const Tensor<T> &src,
//        const std::vector<Index> &src_offset, const Tile<T> &dst,
//        const std::vector<Index> &dst_offset)
//{
//    copy_intersection_async<T>(src, src_offset, dst, dst_offset);
//    starpu_task_wait_for_all();
//}
//
////! Blocking version of tensor-wise copy operation
////
//// @param[in] src: Source tensor
//// @param[inout] dst: Destination tensor
////
//// This operation finds an intersection of the source and the target tensors
//// and copies only the data within the found intersection. No elements of the
//// destination tensor outside the intersection mask are updated. Both the
//// source and the target tensors assumed to have the same offset.
//template<typename T>
//void copy_intersection(const Tensor<T> &src, const Tile<T> &dst)
//{
//    copy_intersection_async<T>(src, std::vector<Index>(src.ndim), dst,
//            std::vector<Index>(dst.ndim));
//    starpu_task_wait_for_all();
//}

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
void copy_intersection_work(const Tensor<T> &src,
        const std::vector<Index> &src_offset, const Tensor<T> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

extern template
void copy_intersection_work(const Tensor<fp32_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp32_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

extern template
void copy_intersection_work(const Tensor<fp64_t> &src,
        const std::vector<Index> &src_offset, const Tensor<fp64_t> &dst,
        const std::vector<Index> &dst_offset,
        const StarpuVariableHandle &scratch);

template<typename T>
void copy_intersection_async(const Tensor<T> &src,
        const std::vector<Index> &src_offset,
        const Tensor<T> &dst, const std::vector<Index> &dst_offset)
{
    // Check dimensions
    if(src.ndim != src_offset.size())
    {
        throw std::runtime_error("src.ndim != src_offset.size()");
    }
    if(src.ndim != dst.ndim)
    {
        throw std::runtime_error("src.ndim != dst.ndim");
    }
    if(dst.ndim != dst_offset.size())
    {
        throw std::runtime_error("dst.ndim != dst_offset.size()");
    }
    // Temporary buffer for indexing, that is allocated per-worker when needed
    StarpuVariableHandle scratch(2 * src.ndim * sizeof(Index));
    // Launch codelet
    copy_intersection_work(src, src_offset, dst, dst_offset, scratch);
}

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

