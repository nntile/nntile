/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/norm_slice.hh
 * TileGraph norm_slice operation
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Norm slice operation: dst = alpha*norm(src1) + beta*src2
struct TileNormSliceOp : TileGraph::OpNode
{
    Scalar alpha = 0, beta = 0;
    Index axis = 0;
    int redux = 0;
    TileGraph::TileNode *s1 = nullptr, *s2 = nullptr, *dst = nullptr;
    TileNormSliceOp() = default;
    TileNormSliceOp(Scalar a, TileGraph::TileNode* t1, Scalar b, TileGraph::TileNode* t2, TileGraph::TileNode* d, Index ax, int r = 0) : alpha(a), beta(b), axis(ax), redux(r), s1(t1), s2(t2), dst(d)
    {
        inputs_ = {s1, s2};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_NORM_SLICE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileNormSliceOp>(*this);
    }
};
void norm_slice(Scalar a, TileGraph::TileNode* t1, Scalar b, TileGraph::TileNode* t2, TileGraph::TileNode* dst, Index ax, int r = 0);
} // namespace nntile::graph::tile_graph
