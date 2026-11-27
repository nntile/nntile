/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/ops/unregister.hh
 * TileGraph async unregister: starpu_data_unregister_submit + drop payload.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tile/graph.hh>

namespace nntile::tile
{

//! Async unregister of a tile handle (StarPU-ordered after last use).
struct TileUnregisterOp : TileGraph::OpNode
{
    TileGraph::TileNode *x = nullptr;

    TileUnregisterOp() = default;
    explicit TileUnregisterOp(TileGraph::TileNode *x_)
        : x(x_)
    {
        // Listed as input so StarPU / DCE see the dependence on prior
        // writers and readers; no outputs (side-effect only).
        inputs_ = {x};
        outputs_ = {};
    }

    std::string op_name() const override { return "TILE_UNREGISTER"; }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileUnregisterOp>(*this);
    }
};

void unregister(TileGraph::TileNode *x);

} // namespace nntile::tile
