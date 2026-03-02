/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add.hh
 * TensorGraph add operation: z = alpha * x + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
{

//! Add operation at tensor level: z = alpha * x + beta * y
struct TensorAddOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    TensorGraph::DataNode* x = nullptr;
    TensorGraph::DataNode* y = nullptr;
    TensorGraph::DataNode* z = nullptr;

    TensorAddOp() = default;
    TensorAddOp(
        TensorGraph::DataNode* x_,
        TensorGraph::DataNode* y_,
        TensorGraph::DataNode* z_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override { return "ADD"; }

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorAddOp>(*this);
    }
};

//! Add operation: z = alpha * x + beta * y
//! @param alpha Scaling factor for x
//! @param x First input tensor
//! @param beta Scaling factor for y
//! @param y Second input tensor
//! @param output_name Name for the output tensor
//! @return Pointer to the output tensor
TensorGraph::DataNode* add(
    Scalar alpha,
    TensorGraph::DataNode* x,
    Scalar beta,
    TensorGraph::DataNode* y,
    const std::string& output_name);

} // namespace nntile::graph
