/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn_graph/add.cc
 * NNGraph add operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/add.hh"
#include "nntile/graph/logical_graph_ops.hh"

#include <stdexcept>

namespace nntile::graph
{

NNGraph::TensorNode* Add::build_forward(
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument("Add::build_forward: x and y must be non-null");
    }
    NNGraph& graph = x->graph();
    LogicalGraph::TensorNode& z_data =
        add(alpha, x->data(), beta, y->data(), output_name);
    bool out_requires_grad = any_input_requires_grad({x, y});
    NNGraph::TensorNode* z = graph.tensor(z_data, out_requires_grad);
    register_op(graph, {x, y}, z, BinaryOpAttrs{alpha, beta},
                [](const NNGraph::OpNode* op) { Add::build_backward(op); });
    return z;
}

void Add::build_backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& attrs = std::get<BinaryOpAttrs>(op->attrs());
    Scalar alpha = attrs.alpha;
    Scalar beta = attrs.beta;
    const auto& inputs = op->inputs();
    if(inputs.size() >= 2 && grad_out != nullptr)
    {
        NNGraph::TensorNode* x_nn = inputs[0];
        NNGraph::TensorNode* y_nn = inputs[1];
        if(x_nn != nullptr && x_nn->requires_grad())
        {
            NNGraph::TensorNode* grad_x =
                graph.get_or_create_grad(x_nn, x_nn->name() + "_grad");
            add_inplace(alpha, grad_out->data(), Scalar(1.0), grad_x->data());
        }
        if(y_nn != nullptr && y_nn->requires_grad())
        {
            NNGraph::TensorNode* grad_y =
                graph.get_or_create_grad(y_nn, y_nn->name() + "_grad");
            add_inplace(beta, grad_out->data(), Scalar(1.0), grad_y->data());
        }
    }
}

} // namespace nntile::graph
