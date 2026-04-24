/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_inplace.cc
 * TensorGraph add_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_inplace.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{



void add_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "add_inplace: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "add_inplace: x and y must be distinct tensors");
    }
    validate_same_shape_and_merge(x, y, "add_inplace");

    auto op = std::make_shared<TensorAddInplaceOp>(x, y, alpha, beta);
    x->graph()->add_op(op);
}

void TensorAddInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vx = tile_lower::tiles_of(m, x);
    const auto& vy = tile_lower::tiles_of(m, y);
    if(vx.size() != vy.size())
    {
        throw std::runtime_error(
            "lower_to_tile ADD_INPLACE: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(x, y, "ADD_INPLACE x/y");
    for(size_t i = 0; i < vx.size(); ++i)
    {
        tile_graph::add_inplace(alpha, vx[i], beta, vy[i]);
    }
}

} // namespace nntile::graph::tensor
