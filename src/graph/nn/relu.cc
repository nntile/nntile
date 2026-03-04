/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/relu.cc
 * NNGraph ReLU autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/relu.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/clear.hh"
#include "nntile/graph/tensor/relu.hh"
#include "nntile/graph/tensor/relu_backward.hh"

namespace nntile::graph
{

NNGraph::TensorNode* NNReluOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNReluOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    TensorGraph::TensorNode* y_data = graph::relu(x->data(), output_name);
    NNGraph::TensorNode* y = graph->tensor(y_data, out_requires_grad);
    outputs_ = {y};
    return y;
}

void NNReluOp::backward() const
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
        if(is_first)
        {
            graph::clear(grad_x->data());
        }
        graph::relu_backward(x->data(), grad_out->data(), grad_x->data());
    }
}

NNGraph::TensorNode* relu(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("relu: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNReluOp>(x);
    NNGraph::TensorNode* y = op->forward(output_name);
    graph->register_op(std::move(op));
    return y;
}

} // namespace nntile::graph
