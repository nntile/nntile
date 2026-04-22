/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/add_slice.hh
 * TileGraph add_slice operation: dst = alpha * src1 + beta * src2
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Add slice operation at tile level: dst = alpha * src1 + beta * src2
struct TileAddSliceOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0;
    TileGraph::TileNode *s1 = nullptr, *s2 = nullptr, *dst = nullptr;
    TileAddSliceOp() = default;
    TileAddSliceOp(Scalar a, TileGraph::TileNode* t1, Scalar b, TileGraph::TileNode* t2, TileGraph::TileNode* d, Index ax) : alpha(a), beta(b), axis(ax), s1(t1), s2(t2), dst(d)
    {
        inputs_ = {s1, s2};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_ADD_SLICE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAddSliceOp>(*this);
    }
};
//! Add slice: dst = alpha * src1 + beta * src2 (uses existing output)
void add_slice(Scalar a, TileGraph::TileNode* t1, Scalar b, TileGraph::TileNode* t2, TileGraph::TileNode* dst, Index axis);
} // namespace nntile::graph::tile_graph
