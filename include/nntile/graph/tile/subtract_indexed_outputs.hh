/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/subtract_indexed_outputs.hh
 * TileGraph subtract_indexed_outputs: dst[labels[i]] -= val
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Subtract indexed outputs: dst[labels[i]] -= val
struct TileSubtractIndexedOutputsOp : TileGraph::OpNode
{
    Scalar v = 0.0;
    Index ignore_index = 0;
    TileGraph::TileNode* labels = nullptr;
    TileGraph::TileNode* dst = nullptr;
    TileSubtractIndexedOutputsOp() = default;
    TileSubtractIndexedOutputsOp(Scalar val, TileGraph::TileNode* l, TileGraph::TileNode* d, Index ig) : v(val), ignore_index(ig), labels(l), dst(d)
    {
        inputs_ = {labels, dst};
        outputs_ = {dst};
    }
    std::string op_name() const override { return "TILE_SUBTRACT_INDEXED_OUTPUTS"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileSubtractIndexedOutputsOp>(*this);
    }
};
//! Subtract indexed outputs: dst[labels[i]] -= val
void subtract_indexed_outputs(Scalar v, TileGraph::TileNode* labels, TileGraph::TileNode* dst, Index ignore_index);
} // namespace nntile::graph::tile_graph
