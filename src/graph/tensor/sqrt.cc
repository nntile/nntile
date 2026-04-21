/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sqrt.cc
 * TensorGraph Sqrt operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sqrt.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sqrt.hh"

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_sqrt(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::sqrt<T>(src_t, dst_t);
}

} // namespace

TensorGraph::TensorNode* sqrt(
    TensorGraph::TensorNode* src,
    const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument("sqrt: input tensor must be non-null");
    }

    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());
    output->set_axes(src->axes());

    auto op = std::make_shared<TensorSqrtOp>(src, output);
    src->graph()->add_op(op);

    return output;
}

void sqrt(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("sqrt: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sqrt: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sqrt: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "sqrt: src and dst must be distinct tensors");
    }
    validate_same_shape_and_merge(src, dst, "sqrt");

    auto op = std::make_shared<TensorSqrtOp>(src, dst);
    src->graph()->add_op(op);
}

void TensorSqrtOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_sqrt<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_sqrt<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_sqrt<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_sqrt<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_sqrt<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            throw std::runtime_error(
                "FP16 not supported for sqrt (use FP32_FAST_FP16)");
            break;
        case DataType::BF16:
            run_sqrt<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for sqrt operation");
        default:
            throw std::runtime_error("Unsupported data type for sqrt");
    }
}

} // namespace nntile::graph::tensor
