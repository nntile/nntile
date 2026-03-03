/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/subtract_indexed_outputs.cc
 * TensorGraph subtract_indexed_outputs operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/subtract_indexed_outputs.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/subtract_indexed_outputs.hh"

namespace nntile::graph
{

namespace
{

template<typename T>
void run_subtract_indexed_outputs(TensorGraph::ExecutionContext& ctx,
                                 Scalar val,
                                 TensorGraph::TensorNode* labels,
                                 TensorGraph::TensorNode* dst,
                                 Index ignore_index)
{
    auto& labels_t = ctx.get_tensor<nntile::int64_t>(labels);
    auto& dst_t = ctx.get_tensor<T>(dst);
    nntile::tensor::subtract_indexed_outputs<T>(
        val, labels_t, dst_t, ignore_index);
}

} // namespace

void subtract_indexed_outputs(Scalar val,
                             TensorGraph::TensorNode* labels,
                             TensorGraph::TensorNode* dst,
                             Index ignore_index)
{
    if(labels == nullptr || dst == nullptr)
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must be non-null");
    if(labels->graph() != dst->graph())
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must belong to same graph");
    if(labels->dtype() != DataType::INT64)
        throw std::invalid_argument(
            "subtract_indexed_outputs: labels must have INT64 dtype");
    auto op = std::make_shared<TensorSubtractIndexedOutputsOp>(
        val, labels, dst, ignore_index);
    dst->graph()->add_op(op);
}

void TensorSubtractIndexedOutputsOp::execute(
    TensorGraph::ExecutionContext& ctx) const
{
    DataType dtype = ctx.get_dtype(dst);
    switch(dtype)
    {
        case DataType::FP32:
            run_subtract_indexed_outputs<nntile::fp32_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::FP32_FAST_TF32:
            run_subtract_indexed_outputs<nntile::fp32_fast_tf32_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::FP32_FAST_FP16:
            run_subtract_indexed_outputs<nntile::fp32_fast_fp16_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::FP32_FAST_BF16:
            run_subtract_indexed_outputs<nntile::fp32_fast_bf16_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::FP64:
            run_subtract_indexed_outputs<nntile::fp64_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::FP16:
            run_subtract_indexed_outputs<nntile::fp16_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::BF16:
            run_subtract_indexed_outputs<nntile::bf16_t>(
                ctx, val, labels, dst, ignore_index);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " not supported for subtract_indexed_outputs");
        default:
            throw std::runtime_error(
                "Unsupported data type for subtract_indexed_outputs");
    }
}

} // namespace nntile::graph
