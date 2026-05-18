/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/log_scalar.hh
 * TileGraph log_scalar: log scalar value from tensor (debugging/monitoring)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Log scalar: log scalar value from tensor
struct TileLogScalarOp : TileGraph::OpNode
{
    std::string name;
    TileGraph::TileNode* value = nullptr;
    TileLogScalarOp() = default;
    TileLogScalarOp(std::string n, TileGraph::TileNode* v) : name(std::move(n)), value(v)
    {
        inputs_ = {value};
        outputs_ = {};
    }
    std::string op_name() const override { return "TILE_LOG_SCALAR"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileLogScalarOp>(*this);
    }
};
//! Log scalar: log scalar value from tensor
void log_scalar(const std::string& name, TileGraph::TileNode* value);
} // namespace nntile::graph::tile_graph
