/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_inplace.hh
 * TensorGraph add_inplace operation: y = alpha * x + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
{

//! Add in-place operation at tensor level: y = alpha * x + beta * y
struct TensorAddInplaceOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    TensorGraph::DataNode* x = nullptr;
    TensorGraph::DataNode* y = nullptr;

    TensorAddInplaceOp() = default;
    TensorAddInplaceOp(
        TensorGraph::DataNode* x_,
        TensorGraph::DataNode* y_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_)
    {
        inputs_ = {x, y};
        outputs_ = {y};
    }

    std::string op_name() const override { return "ADD_INPLACE"; }

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorAddInplaceOp>(*this);
    }
};

//! Add in-place: y = alpha * x + beta * y
void add_inplace(
    Scalar alpha,
    TensorGraph::DataNode* x,
    Scalar beta,
    TensorGraph::DataNode* y);

} // namespace nntile::graph
