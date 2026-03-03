/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/logsumexp.hh
 * TensorGraph logsumexp operation: (src, dst)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! LogSumExp operation: dst = logsumexp(src)
struct TensorLogsumexpOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorLogsumexpOp() = default;
    TensorLogsumexpOp(
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_)
        : src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "LOGSUMEXP"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorLogsumexpOp>(*this);
    }
};

TensorGraph::TensorNode* logsumexp(
    TensorGraph::TensorNode* src,
    const std::string& output_name);

void logsumexp(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph
