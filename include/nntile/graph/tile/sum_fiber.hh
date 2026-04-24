/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sum_fiber.hh
 * TileGraph sum_fiber operation: y = alpha * sum_fiber(x) + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Sum fiber operation at tile level: y = alpha * sum_fiber(x) + beta * y
struct TileSumFiberOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Scalar beta = 0.0;
    Index axis = 0;
    Index batch_ndim = 0;
    int redux = 0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileSumFiberOp() = default;
    TileSumFiberOp(Scalar a, TileGraph::TileNode* s, Scalar b, TileGraph::TileNode* d, Index ax, Index bnd, int r = 0) : alpha(a), beta(b), axis(ax), batch_ndim(bnd), redux(r), src(s), dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SUM_FIBER"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSumFiberOp>(*this);
    }
};
//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y (uses existing output)
void sum_fiber(
    Scalar alpha, TileGraph::TileNode* src, Scalar beta, TileGraph::TileNode* dst, Index axis, Index batch_ndim, int redux = 0);
} // namespace nntile::graph::tile_graph
