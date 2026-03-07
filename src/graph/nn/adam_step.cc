/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/adam_step.cc
 * NNGraph Adam optimizer step implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/adam_step.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

namespace nntile::graph
{

void NNAdamStepOp::forward()
{
    if(param == nullptr || grad == nullptr ||
       first_moment == nullptr || second_moment == nullptr)
    {
        throw std::invalid_argument(
            "NNAdamStepOp::forward: param, grad, first_moment, "
            "second_moment must be non-null");
    }
    graph::tensor::adam_step(
        num_iter, beta_1, beta_2, eps, lr, weight_decay,
        grad->data(), first_moment->data(),
        second_moment->data(), param->data());
}

void NNAdamStepOp::backward() const
{
    throw std::logic_error(
        "NNAdamStepOp::backward: optimizer steps are not differentiable. "
        "Optimizer operations must be used inside a no_grad() scope. "
        "This mirrors PyTorch's torch.no_grad() context for optimizer.step().");
}

void adam_step(
    NNGraph::TensorNode* param,
    NNGraph::TensorNode* grad,
    NNGraph::TensorNode* first_moment,
    NNGraph::TensorNode* second_moment,
    Index num_iter, Scalar beta_1, Scalar beta_2,
    Scalar eps, Scalar lr, Scalar weight_decay)
{
    adam_step(param, grad, first_moment, second_moment,
             std::make_shared<Index>(num_iter), beta_1, beta_2,
             eps, lr, weight_decay);
}

void adam_step(
    NNGraph::TensorNode* param,
    NNGraph::TensorNode* grad,
    NNGraph::TensorNode* first_moment,
    NNGraph::TensorNode* second_moment,
    std::shared_ptr<Index> num_iter, Scalar beta_1, Scalar beta_2,
    Scalar eps, Scalar lr, Scalar weight_decay)
{
    if(param == nullptr || grad == nullptr ||
       first_moment == nullptr || second_moment == nullptr)
    {
        throw std::invalid_argument(
            "adam_step: param, grad, first_moment, second_moment "
            "must be non-null");
    }
    NNGraph* graph = param->graph();
    auto guard = graph->no_grad();

    auto op = std::make_shared<NNAdamStepOp>(
        param, grad, first_moment, second_moment,
        std::move(num_iter), beta_1, beta_2, eps, lr, weight_decay);
    op->forward();
    graph->register_op(std::move(op));
}

} // namespace nntile::graph
