/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/rope.hh
 * TileGraph rope operation: (sin, cos, src, dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! RoPE operation: dst = rope(sin, cos, src)
struct TileRopeOp : TileGraph::OpNode
{
    TileGraph::TileNode* sin = nullptr, * cos = nullptr, * src = nullptr, * dst = nullptr;
    TileRopeOp() = default;
    TileRopeOp(TileGraph::TileNode* si, TileGraph::TileNode* co, TileGraph::TileNode* s, TileGraph::TileNode* d) : sin(si), cos(co), src(s), dst(d)
    {
        inputs_ = {sin, cos, src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_ROPE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileRopeOp>(*this);
    }
};
void rope(TileGraph::TileNode* sin, TileGraph::TileNode* cos, TileGraph::TileNode* src, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
