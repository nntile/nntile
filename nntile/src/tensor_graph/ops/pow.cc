/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/pow.cc
 * TensorGraph pow operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor_graph/ops/pow.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/pow.hh"

#include <nntile/tile_graph/graph_ops.hh>
#include <nntile/tensor_graph/tile_lowering_helpers.hh>

namespace nntile::tensor_graph
{



void pow(
    Scalar alpha,
    Scalar exp,
    TensorGraph::TensorNode* A)
{
    if(A == nullptr)
    {
        throw std::invalid_argument(
            "pow: input tensor must be non-null");
    }

    auto op = std::make_shared<TensorPowOp>(alpha, exp, A);
    A->graph()->add_op(op);
}

void TensorPowOp::lower_to_tile(const LoweringContext& ctx) const
{
    for(TileGraph::TileNode* t : tile_lower::tiles_of(ctx.tile_map, A))
    {
        tile_graph::pow(alpha, exp, t);
    }
}

} // namespace nntile::tensor_graph
