/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/hypot.hh
 * TileGraph hypot operation: (alpha, src1, beta, src2, dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Hypot operation: dst = hypot(alpha*src1, beta*src2)
struct TileHypotOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Scalar beta = 0.0;
    TileGraph::TileNode* src1 = nullptr;
    TileGraph::TileNode* src2 = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileHypotOp() = default;
    TileHypotOp(Scalar a, TileGraph::TileNode* s1, Scalar b, TileGraph::TileNode* s2, TileGraph::TileNode* d)
        : alpha(a), beta(b), src1(s1), src2(s2), dst(d)
    {
        inputs_ = {src1, src2};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_HYPOT"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileHypotOp>(*this);
    }
};
void hypot(
    Scalar alpha, TileGraph::TileNode* src1, Scalar beta, TileGraph::TileNode* src2, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
