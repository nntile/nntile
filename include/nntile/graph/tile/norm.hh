/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/norm.hh
 * TileGraph norm operation: y = alpha * norm(x) + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Norm operation at tile level
struct TileNormOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Scalar beta = 0.0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileNormOp() = default;
    TileNormOp(Scalar a, Scalar b, TileGraph::TileNode* s, TileGraph::TileNode* d)
        : alpha(a), beta(b), src(s), dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_NORM"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileNormOp>(*this);
    }
};
//! Euclidean norm: y = alpha * norm(x) + beta * y
void norm(Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
