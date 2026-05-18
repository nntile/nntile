/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/multiply_inplace.hh
 * TileGraph multiply_inplace operation: dst = alpha * src * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Multiply in-place at tile level: dst = alpha * src * dst
struct TileMultiplyInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0;
    TileGraph::TileNode* src = nullptr, * dst = nullptr;
    TileMultiplyInplaceOp() = default;
    TileMultiplyInplaceOp(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d) : alpha(a), src(s), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_MULTIPLY_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileMultiplyInplaceOp>(*this);
    }
};
//! Multiply in-place: dst = alpha * src * dst
void multiply_inplace(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d);
} // namespace nntile::graph::tile_graph
