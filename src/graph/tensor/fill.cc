/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/fill.cc
 * TensorGraph fill operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/fill.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{

void fill(Scalar val, TensorGraph::TensorNode* x)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("fill: input tensor must be non-null");
    }

    auto op = std::make_shared<TensorFillOp>(x, val);
    x->graph()->add_op(op);
}

void TensorFillOp::lower_to_tile(const LoweringContext& ctx) const
{
    for(TileGraph::TileNode* t : tile_lower::tiles_of(ctx.tile_map, x))
    {
        tile_graph::fill(val, t);
    }
}

} // namespace nntile::graph::tensor
