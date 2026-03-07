/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/adamw_step.hh
 * NNGraph AdamW optimizer step (non-differentiable).
 *
 * Like PyTorch's optimizer.step() under torch.no_grad(), this operation
 * updates parameters in-place and is NOT part of the autograd graph.
 * The backward() method throws std::logic_error as a safety net.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>
#include <nntile/graph/tensor/adamw_step.hh>

namespace nntile::graph
{

struct NNAdamwStepOp : NNGraph::OpNode
{
    Index num_iter;
    Scalar beta_1;
    Scalar beta_2;
    Scalar eps;
    Scalar lr;
    Scalar weight_decay;
    NNGraph::TensorNode* param = nullptr;
    NNGraph::TensorNode* grad = nullptr;
    NNGraph::TensorNode* first_moment = nullptr;
    NNGraph::TensorNode* second_moment = nullptr;

    NNAdamwStepOp() = default;
    NNAdamwStepOp(NNGraph::TensorNode* param_,
                  NNGraph::TensorNode* grad_,
                  NNGraph::TensorNode* first_moment_,
                  NNGraph::TensorNode* second_moment_,
                  Index num_iter_, Scalar beta_1_, Scalar beta_2_,
                  Scalar eps_, Scalar lr_, Scalar weight_decay_)
        : num_iter(num_iter_), beta_1(beta_1_), beta_2(beta_2_),
          eps(eps_), lr(lr_), weight_decay(weight_decay_),
          param(param_), grad(grad_),
          first_moment(first_moment_), second_moment(second_moment_)
    {
        inputs_ = {param, grad, first_moment, second_moment};
    }

    void forward();
    void backward() const override;
};

//! AdamW optimizer step. Automatically uses no_grad scope.
//! Adam with decoupled weight decay.
void adamw_step(
    NNGraph::TensorNode* param,
    NNGraph::TensorNode* grad,
    NNGraph::TensorNode* first_moment,
    NNGraph::TensorNode* second_moment,
    Index num_iter, Scalar beta_1 = 0.9, Scalar beta_2 = 0.999,
    Scalar eps = 1e-8, Scalar lr = 0.001, Scalar weight_decay = 0.01);

} // namespace nntile::graph
