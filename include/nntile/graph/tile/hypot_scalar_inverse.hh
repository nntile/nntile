/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/hypot_scalar_inverse.hh
 * TileGraph hypot_scalar_inverse operation: (eps, alpha, dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Hypot scalar inverse: dst = 1/hypot(alpha*dst, eps)
struct TileHypotScalarInverseOp : TileGraph::OpNode
{
    Scalar eps = 0.0;
    Scalar alpha = 0.0;
    TileGraph::TileNode* dst = nullptr;
    TileHypotScalarInverseOp() = default;
    TileHypotScalarInverseOp(Scalar e, Scalar a, TileGraph::TileNode* d) : eps(e), alpha(a), dst(d)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_HYPOT_SCALAR_INVERSE"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileHypotScalarInverseOp>(*this);
    }
};
void hypot_scalar_inverse(Scalar eps, Scalar alpha, TileGraph::TileNode* dst);
} // namespace nntile::graph::tile_graph
