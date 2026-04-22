/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/gelu.hh
 * TileGraph GeLU: dst = gelu(src)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! GeLU operation at tile level: dst = gelu(src)
struct TileGeluOp : TileGraph::OpNode
{
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;

    TileGeluOp() = default;
    TileGeluOp(TileGraph::TileNode* src_, TileGraph::TileNode* dst_)
        : src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_GELU"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileGeluOp>(*this);
    }
};

//! GeLU activation into pre-created output: dst = gelu(src)
//! @param src Input tile
//! @param dst Output tile (must already exist, same shape as src)
void gelu(TileGraph::TileNode* src, TileGraph::TileNode* dst);

} // namespace nntile::graph::tile_graph
