/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sqrt.hh
 * TileGraph sqrt: dst = sqrt(src)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Sqrt operation at tile level: dst = sqrt(src)
struct TileSqrtOp : TileGraph::OpNode
{
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;

    TileSqrtOp() = default;
    TileSqrtOp(TileGraph::TileNode* src_, TileGraph::TileNode* dst_)
        : src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_SQRT"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSqrtOp>(*this);
    }
};

//! Sqrt: dst = sqrt(src) (uses existing output)
void sqrt(TileGraph::TileNode* src, TileGraph::TileNode* dst);

} // namespace nntile::graph::tile_graph
