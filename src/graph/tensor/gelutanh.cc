/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelutanh.cc
 * TensorGraph gelutanh operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelutanh.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelutanh.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_gelutanh(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    auto& src_t = runtime.get_tensor<T>(src);
    auto& dst_t = runtime.get_tensor<T>(dst);
    nntile::tensor::gelutanh<T>(src_t, dst_t);
}

} // namespace

TensorGraph::TensorNode* gelutanh(
    TensorGraph::TensorNode* src,
    const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument("gelutanh: input tensor must be non-null");
    }

    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode* dst = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());
    dst->set_axes(src->axes());

    gelutanh(src, dst);

    return dst;
}

void gelutanh(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "gelutanh: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "gelutanh: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "gelutanh: src and dst must be distinct tensors");
    }
    validate_same_shape_and_merge(src, dst, "gelutanh");

    auto op = std::make_shared<TensorGelutanhOp>(src, dst);
    src->graph()->add_op(op);
}

void TensorGelutanhOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_gelutanh<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_gelutanh<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_gelutanh<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_gelutanh<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_gelutanh<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            run_gelutanh<nntile::fp16_t>(runtime, src, dst);
            break;
        case DataType::BF16:
            run_gelutanh<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for gelutanh operation");
        default:
            throw std::runtime_error("Unsupported data type for gelutanh");
    }
}

void TensorGelutanhOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_unary2(
        src, dst, ctx.tile_map, "GELUTANH", tile_graph::gelutanh);
}

} // namespace nntile::graph::tensor
