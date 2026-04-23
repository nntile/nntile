/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/relu_backward.cc
 * TensorGraph relu_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/relu_backward.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/tensor.hh"
#include "nntile/tensor/relu_backward.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* relu_backward(TensorGraph::TensorNode* x,
                                       TensorGraph::TensorNode* dy,
                                       const std::string& output_name)
{
    if(x == nullptr || dy == nullptr)
        throw std::invalid_argument("relu_backward: inputs must be non-null");
    if(x->graph() != dy->graph())
        throw std::invalid_argument("relu_backward: inputs must belong to same graph");
    if(x->dtype() != dy->dtype())
        throw std::invalid_argument("relu_backward: inputs must have same dtype");
    if(x == dy)
        throw std::invalid_argument("relu_backward: x and dy must be distinct tensors");
    validate_same_shape_and_merge(x, dy, "relu_backward");
    TensorGraph::TensorNode* output = x->graph()->data(
        x->shape(), output_name, x->dtype());
    output->set_axes(x->axes());
    relu_backward(x, dy, output);
    return output;
}

void relu_backward(TensorGraph::TensorNode* x, TensorGraph::TensorNode* dy,
                   TensorGraph::TensorNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
        throw std::invalid_argument("relu_backward: tensors must be non-null");
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
        throw std::invalid_argument("relu_backward: tensors must belong to same graph");
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
        throw std::invalid_argument("relu_backward: tensors must have same dtype");
    if(x == dy || x == dx || dy == dx)
        throw std::invalid_argument("relu_backward: x, dy, and dx must be distinct tensors");
    validate_same_shape_and_merge(x, dy, "relu_backward");
    validate_same_shape_and_merge(x, dx, "relu_backward");
    auto op = std::make_shared<TensorReluBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorReluBackwardOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vx = tile_lower::tiles_of(m, x);
    const auto& vdy = tile_lower::tiles_of(m, dy);
    const auto& vdx = tile_lower::tiles_of(m, dx);
    if(vx.size() != vdy.size() || vx.size() != vdx.size())
    {
        throw std::runtime_error(
            "lower_to_tile RELU_BACKWARD: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(x, dy, "RELU_BACKWARD x/dy");
    tile_lower::assert_same_elementwise_layout(x, dx, "RELU_BACKWARD x/dx");
    for(size_t i = 0; i < vx.size(); ++i)
    {
        tile_graph::relu_backward(vx[i], vdy[i], vdx[i]);
    }
}

} // namespace nntile::graph::tensor
