/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/add.hh
 * NNGraph add operation - out-of-place z = alpha*x + beta*y.
 *
 * NNAddOp holds params and tensors (like TensorAddOp). Free function add()
 * creates the op and registers it.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/nn_graph.hh>
#include <nntile/graph/nn/op_node.hh>
#include <nntile/graph/tensor/add.hh>

namespace nntile::graph
{

//! Add op: z = alpha*x + beta*y. PyTorch-style: outputs created in forward().
struct NNAddOp : NNGraph::OpNode
{
    Scalar alpha = 1.0;
    Scalar beta = 1.0;
    NNGraph::TensorNode* x = nullptr;
    NNGraph::TensorNode* y = nullptr;

    NNAddOp() = default;
    NNAddOp(NNGraph::TensorNode* x_,
            NNGraph::TensorNode* y_,
            Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_)
    {
        inputs_ = {x, y};
    }

    NNGraph::TensorNode* forward(const std::string& output_name) override;
    void backward() const override;
};

//! Add: z = alpha*x + beta*y. Creates op, adds to graph, registers for backward.
NNGraph::TensorNode* add(
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name);

} // namespace nntile::graph
