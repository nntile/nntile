/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/multiply_slice.hh
 * TileGraph multiply_slice: in-place product with a broadcasted slice
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! TileGraph multiply_slice: dst = alpha * src * dst (slice broadcast)
struct TileMultiplySliceOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Index axis = 0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;

    TileMultiplySliceOp() = default;
    TileMultiplySliceOp(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index ax)
        : alpha(a), axis(ax), src(s), dst(d)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_MULTIPLY_SLICE"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileMultiplySliceOp>(*this);
    }
};

void multiply_slice(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index axis);

} // namespace nntile::graph::tile_graph
