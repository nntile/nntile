/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/add_fiber_inplace.hh
 * TileGraph add_fiber_inplace: tensor = alpha * fiber + beta * tensor
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Add fiber in-place at tile level: tensor = alpha * fiber + beta * tensor
struct TileAddFiberInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0, batch_ndim = 0;
    TileGraph::TileNode* src = nullptr, * dst = nullptr;
    TileAddFiberInplaceOp() = default;
    TileAddFiberInplaceOp(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax, Index bd) : alpha(a), beta(b), axis(ax), batch_ndim(bd), src(s), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_ADD_FIBER_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAddFiberInplaceOp>(*this);
    }
};
//! Add along fibers in-place: tensor = alpha * fiber + beta * tensor
void add_fiber_inplace(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index axis, Index batch_ndim);
} // namespace nntile::graph::tile_graph
