/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/clear.hh
 * TensorGraph clear operation: x = 0
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
{

//! Clear operation at tensor level: x = 0
struct TensorClearOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    TensorGraph::DataNode* x = nullptr;

    TensorClearOp() = default;
    explicit TensorClearOp(TensorGraph::DataNode* x_)
        : x(x_)
    {
        inputs_ = {x};
        outputs_ = {x};
    }

    std::string op_name() const override { return "CLEAR"; }

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorClearOp>(*this);
    }
};

//! Clear tensor: x = 0
void clear(TensorGraph::DataNode* x);

} // namespace nntile::graph
