/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/ops/adam_step.hh
 * TileGraph Adam step for one tile (calls nntile::tile::adam_step).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <memory>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! One tile Adam step.  \p num_iter is fixed when the op is created.
struct TileAdamStepOp : TileGraph::OpNode
{
    Index num_iter = 0;
    Scalar beta_1{};
    Scalar beta_2{};
    Scalar eps{};
    Scalar lr{};
    Scalar weight_decay{};
    TileGraph::TileNode* grad = nullptr;
    TileGraph::TileNode* first_moment = nullptr;
    TileGraph::TileNode* second_moment = nullptr;
    TileGraph::TileNode* p = nullptr;

    TileAdamStepOp() = default;
    TileAdamStepOp(Index num_iter_, Scalar beta_1_, Scalar beta_2_, Scalar eps_,
        Scalar lr_, Scalar weight_decay_, TileGraph::TileNode* grad_,
        TileGraph::TileNode* first_moment_, TileGraph::TileNode* second_moment_,
        TileGraph::TileNode* p_);

    std::string op_name() const override { return "TILE_ADAM_STEP"; }

    void execute(Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileAdamStepOp>(*this);
    }
};

//! Adam step (one tile).
void adam_step(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps,
    Scalar lr, Scalar weight_decay, TileGraph::TileNode* grad,
    TileGraph::TileNode* first_moment, TileGraph::TileNode* second_moment,
    TileGraph::TileNode* p);

} // namespace nntile::graph::tile_graph
