/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/add.cc
 * NNGraph add operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/add.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/add.hh"
#include "nntile/graph/tensor/add_inplace.hh"

namespace nntile::graph
{

namespace
{
constexpr Scalar grad_overwrite = 0.0;
constexpr Scalar grad_accumulate = 1.0;
} // anonymous namespace

NNGraph::TensorNode* NNAddOp::forward(const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "NNAddOp::forward: x, y must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, y});
    NNGraph::TensorNode* z = graph->tensor(
        x->shape(), output_name, x->dtype(), out_requires_grad);
    outputs_ = {z};
    graph::add(alpha, x->data(), beta, y->data(), z->data());
    return z;
}

void NNAddOp::backward() const
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
        graph::add_inplace(alpha, grad_out->data(), grad_beta, grad_x->data());
    }
    if(y != nullptr && y->requires_grad())
    {
        auto [grad_y, is_first] =
            graph->get_or_create_grad(y, y->name() + "_grad");
        Scalar grad_beta = is_first ? grad_overwrite : grad_accumulate;
        graph::add_inplace(beta, grad_out->data(), grad_beta, grad_y->data());
    }
}

NNGraph::TensorNode* add(
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("add: x and y must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNAddOp>(x, y, alpha, beta);
    NNGraph::TensorNode* z = op->forward(output_name);
    register_op(*graph, std::move(op));
    return z;
}

} // namespace nntile::graph
