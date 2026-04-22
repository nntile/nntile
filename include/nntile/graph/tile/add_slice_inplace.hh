/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/add_slice_inplace.hh
 * TileGraph add_slice_inplace operation: dst = alpha * src + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Add slice in-place at tile level: dst = alpha * src + beta * dst
struct TileAddSliceInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0;
    TileGraph::TileNode* src = nullptr, * dst = nullptr;
    TileAddSliceInplaceOp() = default;
    TileAddSliceInplaceOp(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax) : alpha(a), beta(b), axis(ax), src(s), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_ADD_SLICE_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAddSliceInplaceOp>(*this);
    }
};
//! Add slice in-place: dst = alpha * src + beta * dst
void add_slice_inplace(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index axis);
} // namespace nntile::graph::tile_graph
