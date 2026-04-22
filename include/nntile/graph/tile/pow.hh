/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/pow.hh
 * TileGraph pow in-place: A = alpha * A^exp
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

struct TilePowOp : TileGraph::OpNode
{
    Scalar alpha = Scalar{1.0};
    Scalar exp = Scalar{1.0};
    TileGraph::TileNode* A = nullptr;

    TilePowOp() = default;
    TilePowOp(Scalar alpha_, Scalar exp_, TileGraph::TileNode* A_)
        : alpha(alpha_), exp(exp_), A(A_)
    {
        inputs_ = {A};
        outputs_ = {A};
    }

    std::string op_name() const override { return "TILE_POW"; }

    void execute(TileGraph::Runtime& runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TilePowOp>(*this);
    }
};

void pow(Scalar alpha, Scalar exp, TileGraph::TileNode* A);

} // namespace nntile::graph::tile_graph
