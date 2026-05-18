/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sum.hh
 * TileGraph sum operation: dst = alpha * sum(src) + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Sum operation at tile level: dst = alpha * sum(src) + beta * dst
struct TileSumOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Scalar beta = 0.0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileSumOp() = default;
    TileSumOp(Scalar a, Scalar b, TileGraph::TileNode* s, TileGraph::TileNode* d)
        : alpha(a), beta(b), src(s), dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SUM"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSumOp>(*this);
    }
};
//! Sum all elements: dst = alpha * sum(src) + beta * dst
void sum(Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
