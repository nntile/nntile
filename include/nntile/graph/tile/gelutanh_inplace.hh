/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/gelutanh_inplace.hh
 * TileGraph GeLU-tanh in-place
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

struct TileGelutanhInplaceOp : TileGraph::OpNode
{
    TileGraph::TileNode* dst = nullptr;

    TileGelutanhInplaceOp() = default;
    explicit TileGelutanhInplaceOp(TileGraph::TileNode* dst_)
        : dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TILE_GELUTANH_INPLACE"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileGelutanhInplaceOp>(*this);
    }
};

void gelutanh_inplace(TileGraph::TileNode* dst);

} // namespace nntile::graph::tile_graph
