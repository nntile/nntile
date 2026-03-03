/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/gelutanh.cc
 * NNGraph GeLUTanh autograd implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/gelutanh.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <stdexcept>

#include "nntile/graph/tensor/gelutanh.hh"
#include "nntile/graph/tensor/gelutanh_backward.hh"

namespace nntile::graph
{

NNGraph::TensorNode* NNGelutanhOp::forward(const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "NNGelutanhOp::forward: x must be non-null");
    }
    NNGraph* graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x});
    NNGraph::TensorNode* y = graph->tensor(
        x->shape(), output_name, x->dtype(), out_requires_grad);
    outputs_ = {y};
    graph::gelutanh(x->data(), y->data());
    return y;
}

void NNGelutanhOp::backward() const
{
    NNGraph* graph = x->graph();
    NNGraph::TensorNode* grad_out = output()->grad();
    if(grad_out == nullptr)
    {
        return;
    }
    if(x != nullptr && x->requires_grad())
    {
        auto [grad_x, is_first] =
            graph->get_or_create_grad(x, x->name() + "_grad");
        (void)is_first;  // gelutanh_backward overwrites; no beta
        graph::gelutanh_backward(x->data(), grad_out->data(), grad_x->data());
    }
}

NNGraph::TensorNode* gelutanh(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    if(x == nullptr)
    {
        throw std::invalid_argument("gelutanh: x must be non-null");
    }
    NNGraph* graph = x->graph();
    auto op = std::make_shared<NNGelutanhOp>(x);
    NNGraph::TensorNode* y = op->forward(output_name);
    register_op(*graph, std::move(op));
    return y;
}

} // namespace nntile::graph
