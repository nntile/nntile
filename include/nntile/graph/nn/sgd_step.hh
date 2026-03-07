/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/sgd_step.hh
 * NNGraph SGD optimizer step (non-differentiable).
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
#include <nntile/graph/tensor/sgd_step.hh>

namespace nntile::graph
{

struct NNSgdStepOp : NNGraph::OpNode
{
    Index num_iter;
    Scalar momentum;
    Scalar lr;
    Scalar weight_decay;
    Scalar dampening;
    bool nesterov;
    NNGraph::TensorNode* param = nullptr;
    NNGraph::TensorNode* grad = nullptr;
    NNGraph::TensorNode* velocity = nullptr;

    NNSgdStepOp() = default;
    NNSgdStepOp(NNGraph::TensorNode* param_,
                NNGraph::TensorNode* grad_,
                NNGraph::TensorNode* velocity_,
                Index num_iter_, Scalar momentum_, Scalar lr_,
                Scalar weight_decay_, Scalar dampening_, bool nesterov_)
        : num_iter(num_iter_), momentum(momentum_), lr(lr_),
          weight_decay(weight_decay_), dampening(dampening_),
          nesterov(nesterov_),
          param(param_), grad(grad_), velocity(velocity_)
    {
        inputs_ = {param, grad, velocity};
    }

    void forward();
    void backward() const override;
};

//! SGD optimizer step. Automatically uses no_grad scope.
//! p = p - lr * grad (with optional momentum, weight decay, etc.)
void sgd_step(
    NNGraph::TensorNode* param,
    NNGraph::TensorNode* grad,
    NNGraph::TensorNode* velocity,
    Index num_iter, Scalar momentum, Scalar lr,
    Scalar weight_decay = 0.0, Scalar dampening = 0.0,
    bool nesterov = false);

} // namespace nntile::graph
