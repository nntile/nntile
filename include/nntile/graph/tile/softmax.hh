/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/softmax.hh
 * TileGraph softmax operation: (maxsumexp, src, dst, axis)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Softmax operation: dst = softmax(maxsumexp, src, alpha, axis)
struct TileSoftmaxOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Index axis = 0;
    TileGraph::TileNode* maxsumexp = nullptr;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileSoftmaxOp() = default;
    TileSoftmaxOp(
        TileGraph::TileNode* m, TileGraph::TileNode* s, Scalar a, TileGraph::TileNode* d, Index ax)
        : alpha(a), axis(ax), maxsumexp(m), src(s), dst(d)
    {
        inputs_ = {maxsumexp, src};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SOFTMAX"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSoftmaxOp>(*this);
    }
};
void softmax(
    TileGraph::TileNode* maxsumexp_n,
    TileGraph::TileNode* src,
    Scalar alpha,
    TileGraph::TileNode* dst,
    Index axis);
} // namespace nntile::graph::tile_graph
