/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/unregister.cc
 * TileGraph async unregister implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/ops/unregister.hh"

#include <stdexcept>

#include <nntile/runtime.hh>

namespace nntile::tile
{

void unregister(TileGraph::TileNode *x)
{
    if (x == nullptr)
    {
        throw std::invalid_argument(
            "tile unregister: input tile must be non-null");
    }

    auto op = std::make_shared<TileUnregisterOp>(x);
    x->graph()->add_op(op);
}

void TileUnregisterOp::execute(Runtime &runtime) const
{
    runtime.unregister_tile(x);
}

} // namespace nntile::tile
