/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/mse_loss.cc
 * NNGraph mse_loss autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/mse_loss.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add_inplace.hh"
#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/copy.hh"
#include "nntile/graph/tensor/multiply.hh"
#include "nntile/graph/tensor/norm.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
constexpr Scalar beta_fresh = 0.0;
} // anonymous namespace

NNGraph::TensorNode* NNMseLossOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNMseLossOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});

    // norm = ||x|| (scalar)
    const std::string norm_name = output_name + "_norm";
    NNGraph::TensorNode* norm_node =
        graph->tensor({}, norm_name, x->dtype(), false);
    graph::tensor::clear(norm_node->data());
    graph::tensor::norm(x->data(), norm_node->data(), 1.0, beta_fresh);

    // norm_copy for multiply (multiply requires distinct tensors)
    const std::string norm_copy_name = output_name + "_norm_copy";
    TensorGraph::TensorNode* norm_copy_data =
        graph::tensor::copy(norm_node->data(), norm_copy_name);

    // loss = scale * norm^2 = scale * ||x||^2
    TensorGraph::TensorNode* loss_data = graph::tensor::multiply(
        norm_node->data(), norm_copy_data, output_name, scale);

    NNGraph::TensorNode* loss = graph->tensor(loss_data, out_requires_grad);
    outputs_ = {loss};
    buffers_ = {norm_node};
    return loss;
}

void NNMseLossOp::backward() const
{
    NNGraph::TensorNode* out = output();
    if(out == nullptr)
    {
        return;
    }
    NNGraph* graph = out->graph();
    NNGraph::TensorNode* grad_out = out->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, x->name() + "_grad");
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        // grad_x += 2*scale*x (grad_loss implicitly 1.0)
        graph::tensor::add_inplace(2.0 * scale, x->data(), grad_beta,
                                  grad_x->data());
    }
}

NNGraph::TensorNode* mse_loss(
    NNGraph::TensorNode* x,
    const std::string& output_name,
    Scalar scale)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("mse_loss: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNMseLossOp>(x, scale);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
