/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/maxsumexp.hh
 * TensorGraph maxsumexp operation: (src, dst, axis)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! MaxSumExp operation: dst = maxsumexp(src, axis)
struct TensorMaxsumexpOp : TensorGraph::OpNode
{
    Index axis = 0;
    int redux = 0;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorMaxsumexpOp() = default;
    TensorMaxsumexpOp(
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_ = 0)
        : axis(axis_), redux(redux_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "MAXSUMEXP"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMaxsumexpOp>(*this);
    }
};

TensorGraph::TensorNode* maxsumexp(
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    int redux = 0);

void maxsumexp(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux = 0);

} // namespace nntile::graph
