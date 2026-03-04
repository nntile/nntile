/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/copy_intersection.cc
 * TensorGraph copy_intersection operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/copy_intersection.hh"

#include <stdexcept>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/copy_intersection.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_copy_intersection(TensorGraph::Runtime& runtime,
                           TensorGraph::TensorNode* src,
                           const std::vector<Index>& src_offset,
                           TensorGraph::TensorNode* dst,
                           const std::vector<Index>& dst_offset)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::copy_intersection<T>(
        src_t, src_offset, dst_t, dst_offset);
}

} // namespace

void copy_intersection(TensorGraph::TensorNode* src,
                       const std::vector<Index>& src_offset,
                       TensorGraph::TensorNode* dst,
                       const std::vector<Index>& dst_offset)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument(
            "copy_intersection: tensors must be non-null");
    if(src->graph() != dst->graph())
        throw std::invalid_argument(
            "copy_intersection: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument(
            "copy_intersection: tensors must have same dtype");
    if(src_offset.size() != src->ndim() || dst_offset.size() != dst->ndim())
        throw std::invalid_argument(
            "copy_intersection: offset sizes must match tensor ndim");
    auto op = std::make_shared<TensorCopyIntersectionOp>(
        src, src_offset, dst, dst_offset);
    dst->graph()->add_op(op);
}

void TensorCopyIntersectionOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);
    switch(dtype)
    {
        case DataType::FP32:
            run_copy_intersection<nntile::fp32_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::FP32_FAST_TF32:
            run_copy_intersection<nntile::fp32_fast_tf32_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::FP32_FAST_FP16:
            run_copy_intersection<nntile::fp32_fast_fp16_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::FP32_FAST_BF16:
            run_copy_intersection<nntile::fp32_fast_bf16_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::FP64:
            run_copy_intersection<nntile::fp64_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::FP16:
            run_copy_intersection<nntile::fp16_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::BF16:
            run_copy_intersection<nntile::bf16_t>(
                runtime, src, src_offset, dst, dst_offset);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for copy_intersection");
        default:
            throw std::runtime_error(
                "Unsupported data type for copy_intersection");
    }
}

} // namespace nntile::graph::tensor
