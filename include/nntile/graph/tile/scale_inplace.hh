/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/scale_inplace.hh
 * TileGraph scale_inplace operation: dst = alpha * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Scale in-place operation at tile level: dst = alpha * dst
struct TileScaleInplaceOp : TileGraph::OpNode
{
    Scalar alpha = 0.0;
    TileGraph::TileNode* dst = nullptr;
    TileScaleInplaceOp() = default;
    TileScaleInplaceOp(Scalar a, TileGraph::TileNode* d) : alpha(a), dst(d)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SCALE_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileScaleInplaceOp>(*this);
    }
};
//! Scale in-place: dst = alpha * dst
void scale_inplace(Scalar alpha, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
