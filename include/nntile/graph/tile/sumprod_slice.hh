/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sumprod_slice.hh
 * TileGraph sumprod_slice operation: dst = alpha * sumprod_slice(src1, src2) + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Sumprod slice operation: dst = alpha * sumprod_slice(src1, src2) + beta * dst
struct TileSumprodSliceOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0;
    int redux = 0;
    TileGraph::TileNode *s1 = nullptr, *s2 = nullptr, *dst = nullptr;
    TileSumprodSliceOp() = default;
    TileSumprodSliceOp(Scalar a, TileGraph::TileNode* t1, TileGraph::TileNode* t2, Scalar b, TileGraph::TileNode* d, Index ax, int r = 0) : alpha(a), beta(b), axis(ax), redux(r), s1(t1), s2(t2), dst(d)
    {
        inputs_ = {s1, s2};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SUMPROD_SLICE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSumprodSliceOp>(*this);
    }
};
//! Sumprod over fibers into slice: dst = alpha * sumprod_slice(src1, src2) + beta * dst
void sumprod_slice(Scalar a, TileGraph::TileNode* t1, TileGraph::TileNode* t2, Scalar b, TileGraph::TileNode* dst, Index ax, int r = 0);
} // namespace nntile::graph::tile_graph
