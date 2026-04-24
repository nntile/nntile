/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelu.cc
 * TensorGraph GeLU operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelu.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* gelu(
    TensorGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("gelu: input tensor must be non-null");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorGeluOp>(x, output);
    x->graph()->add_op(op);

    return output;
}

void gelu(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("gelu: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "gelu: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "gelu: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "gelu: x and y must be distinct tensors");
    }
    validate_same_shape_and_merge(x, y, "gelu");

    auto op = std::make_shared<TensorGeluOp>(x, y);
    x->graph()->add_op(op);
}

void TensorGeluOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_unary2(
        x, y, ctx.tile_map, "GELU", tile_graph::gelu);
}

} // namespace nntile::graph::tensor
