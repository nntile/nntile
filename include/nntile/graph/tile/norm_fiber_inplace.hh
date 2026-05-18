/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/norm_fiber_inplace.hh
 * TileGraph norm_fiber_inplace operation
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Norm fiber in-place: dst = alpha*norm(src) + beta*dst
struct TileNormFiberInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0, batch_ndim = 0;
    int redux = 0;
    TileGraph::TileNode* src = nullptr, * dst = nullptr;
    TileNormFiberInplaceOp() = default;
    TileNormFiberInplaceOp(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax, Index bd, int r = 0) : alpha(a), beta(b), axis(ax), batch_ndim(bd), redux(r), src(s), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_NORM_FIBER_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileNormFiberInplaceOp>(*this);
    }
};
void norm_fiber_inplace(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax, Index bd, int r = 0);
} // namespace nntile::graph::tile_graph
