/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/relu.cc
 * TensorGraph ReLU operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/relu.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/relu.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_relu(
    TensorGraph::Runtime& runtime,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    nntile::tensor::relu<T>(x_t, y_t);
}

} // namespace

TensorGraph::TensorNode* relu(
    TensorGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("relu: input tensor must be non-null");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorReluOp>(x, output);
    x->graph()->add_op(op);

    return output;
}

void relu(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("relu: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "relu: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "relu: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "relu: x and y must be distinct tensors");
    }
    validate_same_shape_and_merge(x, y, "relu");

    auto op = std::make_shared<TensorReluOp>(x, y);
    x->graph()->add_op(op);
}

void TensorReluOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(src);

    switch(dtype)
    {
        case DataType::FP32:
            run_relu<nntile::fp32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_TF32:
            run_relu<nntile::fp32_fast_tf32_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_FP16:
            run_relu<nntile::fp32_fast_fp16_t>(runtime, src, dst);
            break;
        case DataType::FP32_FAST_BF16:
            run_relu<nntile::fp32_fast_bf16_t>(runtime, src, dst);
            break;
        case DataType::FP64:
            run_relu<nntile::fp64_t>(runtime, src, dst);
            break;
        case DataType::FP16:
            run_relu<nntile::fp16_t>(runtime, src, dst);
            break;
        case DataType::BF16:
            run_relu<nntile::bf16_t>(runtime, src, dst);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                std::string(dtype_to_string(dtype)) +
                " data type not supported for relu operation");
        default:
            throw std::runtime_error("Unsupported data type for relu");
    }
}

void TensorReluOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vs = tile_lower::tiles_of(m, src);
    const auto& vd = tile_lower::tiles_of(m, dst);
    if(vs.size() != vd.size())
    {
        throw std::runtime_error("lower_to_tile RELU: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(src, dst, "RELU src/dst");
    for(size_t i = 0; i < vs.size(); ++i)
    {
        tile_graph::relu(vs[i], vd[i]);
    }
}

} // namespace nntile::graph::tensor
