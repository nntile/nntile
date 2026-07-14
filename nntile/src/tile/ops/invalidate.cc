/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/invalidate.cc
 * TileGraph async invalidate implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/invalidate.hh"

#include <stdexcept>

#include <nntile/runtime.hh>

namespace nntile::tile
{

void invalidate(TileGraph::TileNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "tile invalidate: input tile must be non-null");
    }

    auto op = std::make_shared<TileInvalidateOp>(x);
    x->graph()->add_op(op);
}

void TileInvalidateOp::execute(Runtime &runtime) const
{
    runtime.invalidate_tile(x);
}

} // namespace nntile::tile
