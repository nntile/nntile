/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/gelu_inplace.hh
 * TileGraph GeLU in-place: dst = gelu(dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! GeLU in-place operation: dst = gelu(dst)
struct TileGeluInplaceOp : TileGraph::OpNode
{
    TileGraph::TileNode* dst = nullptr;

    TileGeluInplaceOp() = default;
    explicit TileGeluInplaceOp(TileGraph::TileNode* dst_)
        : dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_GELU_INPLACE"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileGeluInplaceOp>(*this);
    }
};

void gelu_inplace(TileGraph::TileNode* dst);

} // namespace nntile::graph::tile_graph
