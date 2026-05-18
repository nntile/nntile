/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/transpose.hh
 * TileGraph transpose: dst = alpha * transpose(src, ndim)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Transpose operation at tile level: dst = alpha * transpose(src)
struct TileTransposeOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    Index ndim = 0;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* dst = nullptr;

    TileTransposeOp() = default;
    TileTransposeOp(Scalar a, TileGraph::TileNode* s, TileGraph::TileNode* d, Index n)
        : alpha(a)
        , ndim(n)
        , src(s)
        , dst(d)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_TRANSPOSE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTransposeOp>(*this);
    }
};

//! Transpose: dst = alpha * transpose(src) (uses existing output)
void transpose(Scalar alpha, TileGraph::TileNode* src, TileGraph::TileNode* dst, Index ndim);

} // namespace nntile::graph::tile_graph
