/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tile/ops/swap_two_axes.hh
 * TileGraph swap_two_axes: swap two tensor axes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/tile/graph.hh>

namespace nntile::tile
{

struct TileSwapTwoAxesOp : TileGraph::OpNode
{
    Index dim0 = 0;
    Index dim1 = 0;
    TileGraph::TileNode *src = nullptr;
    TileGraph::TileNode *dst = nullptr;

    TileSwapTwoAxesOp() = default;
    TileSwapTwoAxesOp(
        TileGraph::TileNode *src_,
        TileGraph::TileNode *dst_,
        Index dim0_,
        Index dim1_) :
        dim0(dim0_),
        dim1(dim1_),
        src(src_),
        dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_SWAP_TWO_AXES"; }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSwapTwoAxesOp>(*this);
    }
};

void swap_two_axes(
    TileGraph::TileNode *src,
    TileGraph::TileNode *dst,
    Index dim0,
    Index dim1);

} // namespace nntile::tile
