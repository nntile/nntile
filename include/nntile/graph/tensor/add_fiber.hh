/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_fiber.hh
 * TensorGraph add_fiber operation: output = alpha * fiber + beta * tensor
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor_graph.hh>
#include <nntile/graph/base_op_node.hh>

namespace nntile::graph
{

//! Add fiber operation at tensor level: output = alpha * fiber + beta * tensor
struct TensorAddFiberOp : BaseOpNode<TensorGraph, TensorGraphNode>
{
    Index axis = 0;
    Index batch_ndim = 0;
    Scalar alpha = 1.0;
    Scalar beta = 1.0;
    TensorGraph::DataNode* fiber = nullptr;
    TensorGraph::DataNode* tensor = nullptr;
    TensorGraph::DataNode* output = nullptr;

    TensorAddFiberOp() = default;
    TensorAddFiberOp(
        TensorGraph::DataNode* fiber_,
        TensorGraph::DataNode* tensor_,
        TensorGraph::DataNode* output_,
        Scalar alpha_, Scalar beta_,
        Index axis_, Index batch_ndim_)
        : axis(axis_), batch_ndim(batch_ndim_)
        , alpha(alpha_), beta(beta_)
        , fiber(fiber_), tensor(tensor_), output(output_)
    {
        inputs_ = {fiber, tensor};
        outputs_ = {output};
    }

    std::string op_name() const override { return "ADD_FIBER"; }

    void execute(ExecutionContext<TensorGraph::DataNode>& ctx) const override;

    std::shared_ptr<BaseOpNode<TensorGraph, TensorGraphNode>> clone() const override
    {
        return std::make_shared<TensorAddFiberOp>(*this);
    }
};

//! Add along fibers: output = alpha * fiber + beta * tensor (creates output)
TensorGraph::DataNode* add_fiber(
    Scalar alpha,
    TensorGraph::DataNode* fiber,
    Scalar beta,
    TensorGraph::DataNode* tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim = 0);

//! Add along fibers: output = alpha * fiber + beta * tensor (uses existing output)
void add_fiber(
    Scalar alpha,
    TensorGraph::DataNode* fiber,
    Scalar beta,
    TensorGraph::DataNode* tensor,
    TensorGraph::DataNode* output,
    Index axis,
    Index batch_ndim = 0);

} // namespace nntile::graph
