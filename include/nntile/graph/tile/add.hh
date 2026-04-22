/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/add.hh
 * TileGraph add operation: z = alpha * x + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Add operation at tile level: z = alpha * x + beta * y
struct TileAddOp : TileGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TileGraph::TileNode* x = nullptr;
    TileGraph::TileNode* y = nullptr;
    TileGraph::TileNode* z = nullptr;

    TileAddOp() = default;
    TileAddOp(
        TileGraph::TileNode* x_,
        TileGraph::TileNode* y_,
        TileGraph::TileNode* z_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override { return "TILE_ADD"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAddOp>(*this);
    }
};

//! Add operation: z = alpha * x + beta * y (creates output)
//! @param alpha Scaling factor for x
//! @param x First input tile
//! @param beta Scaling factor for y
//! @param y Second input tile
//! @param output_name Name for the output tile
//! @return Pointer to the output tile
TileGraph::TileNode* add(
    Scalar alpha,
    TileGraph::TileNode* x,
    Scalar beta,
    TileGraph::TileNode* y,
    const std::string& output_name);

//! Add operation: z = alpha * x + beta * y (uses existing output)
void add(
    Scalar alpha,
    TileGraph::TileNode* x,
    Scalar beta,
    TileGraph::TileNode* y,
    TileGraph::TileNode* z);

} // namespace nntile::graph::tile_graph
