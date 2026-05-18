/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/gelutanh_backward.hh
 * TileGraph GeLU-tanh backward
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! GeLUTanh backward operation: dx = gelutanh_backward(x, dy)
struct TileGelutanhBackwardOp : TileGraph::OpNode
{
    TileGraph::TileNode* x = nullptr;
    TileGraph::TileNode* dy = nullptr;
    TileGraph::TileNode* dx = nullptr;

    TileGelutanhBackwardOp() = default;
    TileGelutanhBackwardOp(
        TileGraph::TileNode* x_, TileGraph::TileNode* dy_, TileGraph::TileNode* dx_)
        : x(x_), dy(dy_), dx(dx_)
    {
        inputs_ = {x, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "TILE_GELUTANH_BACKWARD"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileGelutanhBackwardOp>(*this);
    }
};

void gelutanh_backward(
    TileGraph::TileNode* x, TileGraph::TileNode* dy, TileGraph::TileNode* dx);

} // namespace nntile::graph::tile_graph
