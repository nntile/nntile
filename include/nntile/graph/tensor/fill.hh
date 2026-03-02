/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/fill.hh
 * TensorGraph fill operation: x = val
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
{

//! Fill operation at tensor level: x = val
struct TensorFillOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    Scalar val = 0.0;
    TensorGraph::DataNode* x = nullptr;

    TensorFillOp() = default;
    TensorFillOp(TensorGraph::DataNode* x_, Scalar val_)
        : val(val_), x(x_)
    {
        inputs_ = {x};
        outputs_ = {x};
    }

    std::string op_name() const override { return "FILL"; }

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorFillOp>(*this);
    }
};

//! Fill tensor: x = val
void fill(Scalar val, TensorGraph::DataNode* x);

} // namespace nntile::graph
