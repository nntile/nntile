/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/sgd_step.cc
 * NNGraph SGD optimizer step implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/sgd_step.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

namespace nntile::graph
{

void NNSgdStepOp::forward()
{
    if(param == nullptr || grad == nullptr || velocity == nullptr)
    {
        throw std::invalid_argument(
            "NNSgdStepOp::forward: param, grad, velocity must be non-null");
    }
    graph::tensor::sgd_step(
        num_iter, momentum, lr, weight_decay, dampening, nesterov,
        grad->data(), velocity->data(), param->data());
}

void NNSgdStepOp::backward() const
{
    throw std::logic_error(
        "NNSgdStepOp::backward: optimizer steps are not differentiable. "
        "Optimizer operations must be used inside a no_grad() scope. "
        "This mirrors PyTorch's torch.no_grad() context for optimizer.step().");
}

void sgd_step(
    NNGraph::TensorNode* param,
    NNGraph::TensorNode* grad,
    NNGraph::TensorNode* velocity,
    Index num_iter, Scalar momentum, Scalar lr,
    Scalar weight_decay, Scalar dampening, bool nesterov)
{
    sgd_step(param, grad, velocity,
             std::make_shared<Index>(num_iter), momentum, lr,
             weight_decay, dampening, nesterov);
}

void sgd_step(
    NNGraph::TensorNode* param,
    NNGraph::TensorNode* grad,
    NNGraph::TensorNode* velocity,
    std::shared_ptr<Index> num_iter, Scalar momentum, Scalar lr,
    Scalar weight_decay, Scalar dampening, bool nesterov)
{
    if(param == nullptr || grad == nullptr || velocity == nullptr)
    {
        throw std::invalid_argument(
            "sgd_step: param, grad, velocity must be non-null");
    }
    NNGraph* graph = param->graph();
    auto guard = graph->no_grad();

    auto op = std::make_shared<NNSgdStepOp>(
        param, grad, velocity,
        std::move(num_iter), momentum, lr, weight_decay, dampening, nesterov);
    op->forward();
    graph->register_op(std::move(op));
}

} // namespace nntile::graph
