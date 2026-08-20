/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/unregister.cc
 * TensorGraph async unregister implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/unregister.hh"

#include <stdexcept>

#include "nntile/tensor.hh"
#include "nntile/tile/graph_ops.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"

namespace nntile::tensor
{

void unregister(TensorGraph::TensorNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "unregister: tensor must be non-null");
    }

    auto op = std::make_shared<TensorUnregisterOp>(x);
    x->graph()->add_op(op);
}

void TensorUnregisterOp::lower_to_tile(const LoweringContext &ctx) const
{
    for (TileGraph::TileNode *t : tile_lower::tiles_of(ctx.tile_map, x))
    {
        tile::unregister(t);
    }
}

} // namespace nntile::tensor
