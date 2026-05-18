/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/maxsumexp.hh
 * TileGraph maxsumexp operation: (src, dst, axis)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! MaxSumExp operation: dst = maxsumexp(src, axis)
struct TileMaxsumexpOp : TileGraph::OpNode
{
    Index axis = 0;
    int redux = 0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileMaxsumexpOp() = default;
    TileMaxsumexpOp(TileGraph::TileNode* s, TileGraph::TileNode* d, Index ax, int r = 0) : axis(ax), redux(r), src(s), dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_MAXSUMEXP"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileMaxsumexpOp>(*this);
    }
};
void maxsumexp(TileGraph::TileNode* src, TileGraph::TileNode* dst, Index axis, int redux = 0);
} // namespace nntile::graph::tile_graph
