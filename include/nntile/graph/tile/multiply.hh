/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/multiply.hh
 * TileGraph multiply: z = alpha * x * y
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Multiply operation at tile level
struct TileMultiplyOp : TileGraph::OpNode
{
    Scalar alpha = Scalar{1.0};
    TileGraph::TileNode* x = nullptr;
    TileGraph::TileNode* y = nullptr;
    TileGraph::TileNode* z = nullptr;

    TileMultiplyOp() = default;
    TileMultiplyOp(
        Scalar alpha_,
        TileGraph::TileNode* x_,
        TileGraph::TileNode* y_,
        TileGraph::TileNode* z_)
        : alpha(alpha_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override { return "TILE_MULTIPLY"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileMultiplyOp>(*this);
    }
};

void multiply(
    Scalar alpha,
    TileGraph::TileNode* x,
    TileGraph::TileNode* y,
    TileGraph::TileNode* z);

} // namespace nntile::graph::tile_graph
