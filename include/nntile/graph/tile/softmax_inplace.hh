/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/softmax_inplace.hh
 * TileGraph softmax_inplace operation: (maxsumexp, alpha, dst, axis)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Softmax in-place: dst = softmax(maxsumexp, alpha, dst, axis)
struct TileSoftmaxInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Index axis = 0;
    TileGraph::TileNode* maxsumexp_n = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileSoftmaxInplaceOp() = default;
    TileSoftmaxInplaceOp(TileGraph::TileNode* m, Scalar a, TileGraph::TileNode* d, Index ax) : alpha(a), axis(ax), maxsumexp_n(m), dst(d)
    {
        inputs_ = {maxsumexp_n, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SOFTMAX_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSoftmaxInplaceOp>(*this);
    }
};
void softmax_inplace(TileGraph::TileNode* mse, Scalar alpha, TileGraph::TileNode* dst, Index axis);
} // namespace nntile::graph::tile_graph
