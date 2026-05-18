/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add.cc
 * TensorGraph add operation implementation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/tensor/add.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor.hh>
#include <nntile/tensor/add.hh>

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* add(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("add: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "add: input tensors must belong to the same graph");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "add: x and y must be distinct tensors");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(x, y, "add");

    TensorGraph::TensorNode* output = x->graph()->data(
        x->shape(), output_name, x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorAddOp>(x, y, output, alpha, beta);
    x->graph()->add_op(op);

    return output;
}

void add(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y,
    TensorGraph::TensorNode* z)
{
    if(x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument("add: input tensors must be non-null");
    }
    if(x == y || x == z || y == z)
    {
        throw std::invalid_argument(
            "add: x, y, and z must be distinct tensors");
    }
    if(x->graph() != y->graph() || x->graph() != z->graph())
    {
        throw std::invalid_argument(
            "add: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype() || x->dtype() != z->dtype())
    {
        throw std::invalid_argument(
            "add: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(x, y, "add");
    validate_same_shape_and_merge(x, z, "add");

    auto op = std::make_shared<TensorAddOp>(x, y, z, alpha, beta);
    x->graph()->add_op(op);
}

void TensorAddOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vx = tile_lower::tiles_of(m, x);
    const auto& vy = tile_lower::tiles_of(m, y);
    const auto& vz = tile_lower::tiles_of(m, z);
    if(vx.size() != vy.size() || vx.size() != vz.size())
    {
        throw std::runtime_error(
            "lower_to_tile ADD: tile count mismatch for operands");
    }
    tile_lower::assert_same_elementwise_layout(x, y, "ADD x/y");
    tile_lower::assert_same_elementwise_layout(x, z, "ADD x/z");
    for(size_t i = 0; i < vx.size(); ++i)
    {
        tile_graph::add(alpha, vx[i], beta, vy[i], vz[i]);
    }
}

} // namespace nntile::graph::tensor
