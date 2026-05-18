/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/fill.hh
 * TileGraph fill operation: x = val
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Fill operation at tile level: x = val
struct TileFillOp : TileGraph::OpNode
{
    Scalar val;
    TileGraph::TileNode* x = nullptr;

    TileFillOp() = default;
    TileFillOp(TileGraph::TileNode* x_, Scalar val_)
        : val(val_), x(x_)
    {
        inputs_ = {};
        outputs_ = {x};
    }

    std::string op_name() const override { return "TILE_FILL"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileFillOp>(*this);
    }
};

//! Fill tile: x = val
void fill(Scalar val, TileGraph::TileNode* x);

} // namespace nntile::graph::tile_graph
