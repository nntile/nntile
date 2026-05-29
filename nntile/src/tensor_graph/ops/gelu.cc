#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/gelu.cc
 * TensorGraph GeLU operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor_graph/ops/gelu.hh"

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/gelu.hh"

#include <nntile/tensor_graph/tile_lowering_helpers.hh>
#include <nntile/tile_graph/graph_ops.hh>
#include <stdexcept>
#include <utility>

namespace nntile::tensor_graph
{

TensorGraph::TensorNode *gelu(TensorGraph::TensorNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument("gelu: input tensor must be non-null");
    }

    std::vector<Index> output_shape = x->shape();
    TensorGraph::TensorNode *output =
        x->graph()->data(std::move(output_shape), x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorGeluOp>(x, output);
    x->graph()->add_op(op);

    return output;
}

void gelu(TensorGraph::TensorNode *x, TensorGraph::TensorNode *y)
{
    if (x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("gelu: input tensors must be non-null");
    }
    if (x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "gelu: input tensors must belong to the same graph");
    }
    if (x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "gelu: input tensors must have the same dtype");
    }
    if (x == y)
    {
        throw std::invalid_argument("gelu: x and y must be distinct tensors");
    }
    validate_same_shape_and_merge(x, y, "gelu");

    auto op = std::make_shared<TensorGeluOp>(x, y);
    x->graph()->add_op(op);
}

void TensorGeluOp::lower_to_tile(const LoweringContext &ctx) const
{
    tile_lower::lower_unary2(x, y, ctx.tile_map, "GELU", tile_graph::gelu);
}

} // namespace nntile::tensor_graph
