/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sqrt_inplace.hh
 * TileGraph sqrt_inplace operation: dst = sqrt(dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! ReLU in-place at tile level: dst = sqrt(dst)
struct TileSqrtInplaceOp : TileGraph::OpNode
{
    TileGraph::TileNode* dst = nullptr;
    TileSqrtInplaceOp() = default;
    explicit TileSqrtInplaceOp(TileGraph::TileNode* d) : dst(d)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SQRT_INPLACE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSqrtInplaceOp>(*this);
    }
};
//! ReLU in-place: dst = sqrt(dst)
void sqrt_inplace(TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
