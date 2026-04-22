/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply.cc
 * TensorGraph multiply operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/multiply.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{

namespace
{

template<typename T>
void run_multiply(
    TensorGraph::Runtime& runtime,
    Scalar alpha,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    TensorGraph::TensorNode* z)
{
    auto& x_t = runtime.get_tensor<T>(x);
    auto& y_t = runtime.get_tensor<T>(y);
    auto& z_t = runtime.get_tensor<T>(z);
    nntile::tensor::multiply<T>(alpha, x_t, y_t, z_t);
}

} // namespace

TensorGraph::TensorNode* multiply(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    const std::string& output_name,
    Scalar alpha)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "multiply: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "multiply: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "multiply: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "multiply: x and y must be distinct tensors");
    }
    validate_same_shape_and_merge(x, y, "multiply");

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorMultiplyOp>(x, y, output, alpha);
    x->graph()->add_op(op);

    return output;
}

void TensorMultiplyOp::execute(
    TensorGraph::Runtime& runtime) const
{
    DataType dtype = runtime.get_dtype(x);

    switch(dtype)
    {
        case DataType::FP32:
            run_multiply<nntile::fp32_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_TF32:
            run_multiply<nntile::fp32_fast_tf32_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_FP16:
            run_multiply<nntile::fp32_fast_fp16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP32_FAST_BF16:
            run_multiply<nntile::fp32_fast_bf16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP64:
            run_multiply<nntile::fp64_t>(runtime, alpha, x, y, z);
            break;
        case DataType::FP16:
            run_multiply<nntile::fp16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::BF16:
            run_multiply<nntile::bf16_t>(runtime, alpha, x, y, z);
            break;
        case DataType::INT64:
        case DataType::BOOL:
            throw std::runtime_error(
                "INT64/BOOL data type not supported for multiply operation");
        default:
            throw std::runtime_error("Unsupported data type for multiply");
    }
}

void TensorMultiplyOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vx = tile_lower::tiles_of(m, x);
    const auto& vy = tile_lower::tiles_of(m, y);
    const auto& vz = tile_lower::tiles_of(m, z);
    if(vx.size() != vy.size() || vx.size() != vz.size())
    {
        throw std::runtime_error(
            "lower_to_tile MULTIPLY: tile count mismatch for operands");
    }
    tile_lower::assert_same_elementwise_layout(x, y, "MULTIPLY x/y");
    tile_lower::assert_same_elementwise_layout(x, z, "MULTIPLY x/z");
    for(size_t i = 0; i < vx.size(); ++i)
    {
        tile_graph::multiply(alpha, vx[i], vy[i], vz[i]);
    }
}

} // namespace nntile::graph::tensor
