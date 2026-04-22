/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/total_sum_accum.hh
 * TileGraph total_sum_accum operation
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Total sum accumulating: (alpha, logsumexp, src, class_labels, val, ignore_index)
struct TileTotalSumAccumOp : TileGraph::OpNode
{
    Scalar alpha = 0;
    Index ignore_index = 0;
    TileGraph::TileNode* logsumexp = nullptr;
    TileGraph::TileNode* src = nullptr;
    TileGraph::TileNode* class_labels = nullptr;
    TileGraph::TileNode* val = nullptr; // FP32
    TileTotalSumAccumOp() = default;
    TileTotalSumAccumOp(Scalar a, TileGraph::TileNode* l, TileGraph::TileNode* s, TileGraph::TileNode* cl, TileGraph::TileNode* v, Index ig) : alpha(a), ignore_index(ig), logsumexp(l), src(s), class_labels(cl), val(v)
    {
        inputs_ = {logsumexp, src, class_labels, val};
        outputs_ = {val};
    }
    std::string op_name() const override { return "TILE_TOTAL_SUM_ACCUM"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTotalSumAccumOp>(*this);
    }
};
void total_sum_accum(Scalar a, TileGraph::TileNode* lse, TileGraph::TileNode* src, TileGraph::TileNode* labels, TileGraph::TileNode* val, Index ignore_index);
} // namespace nntile::graph::tile_graph
