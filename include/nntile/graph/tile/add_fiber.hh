/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/add_fiber.hh
 * TileGraph add_fiber operation: output = alpha * fiber + beta * tensor
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Add fiber operation at tile level: output = alpha * fiber + beta * tensor
struct TileAddFiberOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Scalar beta = 0.0;
    Index axis = 0;
    Index batch_ndim = 0;
    TileGraph::TileNode* s1 = nullptr;
    TileGraph::TileNode* s2 = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileAddFiberOp() = default;
    TileAddFiberOp(
        Scalar a, TileGraph::TileNode* a1, Scalar b, TileGraph::TileNode* a2, TileGraph::TileNode* d, Index ax, Index bd)
        : alpha(a), beta(b), axis(ax), batch_ndim(bd), s1(a1), s2(a2), dst(d)
    {
        inputs_ = {s1, s2};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_ADD_FIBER"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAddFiberOp>(*this);
    }
};
//! Add along fibers: output = alpha * fiber + beta * tensor (uses existing output)
void add_fiber(Scalar a, TileGraph::TileNode* s1, Scalar b, TileGraph::TileNode* s2, TileGraph::TileNode* dst, Index axis, Index batch_ndim);
} // namespace nntile::graph::tile_graph
