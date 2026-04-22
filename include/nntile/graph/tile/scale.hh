/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/scale.hh
 * TileGraph scale operation: dst = alpha * src
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Scale operation at tile level: dst = alpha * src
struct TileScaleOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileScaleOp() = default;
    TileScaleOp(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d) : alpha(a), src(s), dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SCALE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileScaleOp>(*this);
    }
};
//! Scale operation: dst = alpha * src (uses existing output)
void scale(Scalar alpha, TileGraph::TileNode* src, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
