/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/add_inplace.hh
 * TileGraph add_inplace operation: y = alpha * x + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Add in-place operation at tile level: y = alpha * x + beta * y
struct TileAddInplaceOp : TileGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TileGraph::TileNode* x = nullptr;
    TileGraph::TileNode* y = nullptr;

    TileAddInplaceOp() = default;
    TileAddInplaceOp(
        TileGraph::TileNode* x_,
        TileGraph::TileNode* y_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_)
    {
        inputs_ = {x, y};
        outputs_ = {y};
    }

    std::string op_name() const override { return "TILE_ADD_INPLACE"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAddInplaceOp>(*this);
    }
};

//! Add in-place: y = alpha * x + beta * y
void add_inplace(
    Scalar alpha,
    TileGraph::TileNode* x,
    Scalar beta,
    TileGraph::TileNode* y);

} // namespace nntile::graph::tile_graph
