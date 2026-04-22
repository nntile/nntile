/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/sgd_step.hh
 * TileGraph SGD step for one tile (calls nntile::tile::sgd_step).
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>

#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

struct TileSgdStepOp : TileGraph::OpNode
{
    std::shared_ptr<Index> step_iter;
    bool bump_after = false;
    Scalar momentum{};
    Scalar lr{};
    Scalar weight_decay{};
    Scalar dampening{};
    bool nesterov = false;
    TileGraph::TileNode* grad = nullptr;
    TileGraph::TileNode* velocity = nullptr;
    TileGraph::TileNode* p = nullptr;

    TileSgdStepOp() = default;
    TileSgdStepOp(const std::shared_ptr<Index>& step_iter_, bool bump_after_,
        Scalar momentum_, Scalar lr_, Scalar weight_decay_, Scalar dampening_,
        bool nesterov_, TileGraph::TileNode* grad_, TileGraph::TileNode* velocity_,
        TileGraph::TileNode* p_);

    std::string op_name() const override { return "TILE_SGD_STEP"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSgdStepOp>(*this);
    }
};

void sgd_step(const std::shared_ptr<Index>& step_iter, bool bump_after,
    Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening,
    bool nesterov, TileGraph::TileNode* grad, TileGraph::TileNode* velocity,
    TileGraph::TileNode* p);

} // namespace nntile::graph::tile_graph
