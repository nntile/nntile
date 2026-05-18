/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/mask_scalar.hh
 * TileGraph mask_scalar operation: A[mask] = val
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Mask scalar operation: A[mask] = val
struct TileMaskScalarOp : TileGraph::OpNode
{
    Scalar val = 0.0;
    Index batch_ndim = 0;
    TileGraph::TileNode* mask = nullptr;
    TileGraph::TileNode* a = nullptr;
    TileMaskScalarOp() = default;
    TileMaskScalarOp(
        TileGraph::TileNode* m, Scalar v, TileGraph::TileNode* a_, Index b) : val(v), batch_ndim(b), mask(m), a(a_)
    {
        inputs_ = {mask, a};
        outputs_ = {a};
    }
    std::string op_name() const override { return "TILE_MASK_SCALAR"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileMaskScalarOp>(*this);
    }
};
//! Mask scalar: A[mask] = val
void mask_scalar(
    TileGraph::TileNode* mask, Scalar val, TileGraph::TileNode* a, Index batch_ndim = 0);
} // namespace nntile::graph::tile_graph
