/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sumprod_fiber.hh
 * TileGraph sumprod_fiber operation: dst = alpha * sumprod_fiber(src1, src2) + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Sumprod fiber operation: dst = alpha * sumprod_fiber(src1, src2) + beta * dst
struct TileSumprodFiberOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0;
    int redux = 0;
    TileGraph::TileNode *s1 = nullptr, *s2 = nullptr, *dst = nullptr;
    TileSumprodFiberOp() = default;
    TileSumprodFiberOp(Scalar a, TileGraph::TileNode* a1, TileGraph::TileNode* a2, Scalar b, TileGraph::TileNode* d, Index ax, int r = 0) : alpha(a), beta(b), axis(ax), redux(r), s1(a1), s2(a2), dst(d)
    {
        inputs_ = {s1, s2};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SUMPROD_FIBER"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSumprodFiberOp>(*this);
    }
};
//! Sumprod over fibers: dst = alpha * sumprod_fiber(src1, src2) + beta * dst
void sumprod_fiber(Scalar alpha, TileGraph::TileNode* a, TileGraph::TileNode* b, Scalar beta, TileGraph::TileNode* dst, Index axis, int redux = 0);
} // namespace nntile::graph::tile_graph
