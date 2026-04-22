/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/hypot_inplace.hh
 * TileGraph hypot_inplace operation: (alpha, src, beta, dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Hypot in-place: dst = hypot(alpha*src, beta*dst)
struct TileHypotInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Scalar beta = 0.0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileHypotInplaceOp() = default;
    TileHypotInplaceOp(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d)
        : alpha(a), src(s), beta(b), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_HYPOT_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileHypotInplaceOp>(*this);
    }
};
void hypot_inplace(Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
