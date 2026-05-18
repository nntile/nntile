/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/scale_fiber.hh
 * TileGraph scale_fiber operation: dst = alpha * src (fiber broadcast)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Scale fiber operation: dst = alpha * src (fiber broadcast)
struct TileScaleFiberOp : TileGraph::OpNode
{
    Scalar alpha = 0;
    Index axis = 0, batch_ndim = 0;
    TileGraph::TileNode* src = nullptr, * dst = nullptr;
    TileScaleFiberOp() = default;
    TileScaleFiberOp(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index ax, Index bd) : alpha(a), axis(ax), batch_ndim(bd), src(s), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SCALE_FIBER"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileScaleFiberOp>(*this);
    }
};
void scale_fiber(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index axis, Index batch_ndim);
} // namespace nntile::graph::tile_graph
