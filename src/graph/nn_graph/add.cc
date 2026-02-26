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
#include "nntile/graph/nn_graph_backward.hh"
#include "nntile/graph/logical_graph_ops.hh"

namespace nntile::graph
{

NNGraph::TensorNode& Add::build_forward(
    Scalar alpha,
    NNGraph::TensorNode& x,
    Scalar beta,
    NNGraph::TensorNode& y,
    const std::string& output_name)
{
    NNGraph& graph = x.graph();
    LogicalGraph::TensorNode& z_data =
        add(alpha, x.data(), beta, y.data(), output_name);
    bool out_requires_grad = x.requires_grad() || y.requires_grad();
    return graph.tensor(z_data, out_requires_grad);
}

void Add::build_backward(
    NNGraph& graph,
    LogicalGraph::OpNode* op,
    NNGraph::TensorNode* grad_out)
{
    const auto& attrs = std::get<BinaryOpAttrs>(op->attrs());
    Scalar alpha = attrs.alpha;
    Scalar beta = attrs.beta;
    if(op->inputs().size() >= 2)
    {
        NNGraph::TensorNode* x_nn = graph.get_tensor(op->input(0)->name());
        NNGraph::TensorNode* y_nn = graph.get_tensor(op->input(1)->name());
        if(x_nn != nullptr && x_nn->requires_grad())
        {
            NNGraph::TensorNode& grad_x =
                graph.get_or_create_grad(*x_nn, x_nn->name() + "_grad");
            add_inplace(alpha, grad_out->data(), Scalar(1.0), grad_x.data());
        }
        if(y_nn != nullptr && y_nn->requires_grad())
        {
            NNGraph::TensorNode& grad_y =
                graph.get_or_create_grad(*y_nn, y_nn->name() + "_grad");
            add_inplace(beta, grad_out->data(), Scalar(1.0), grad_y.data());
        }
    }
}

namespace
{
struct RegisterAddBackward
{
    RegisterAddBackward()
    {
        register_backward(OpType::ADD, Add::build_backward);
    }
} register_add_backward;
} // namespace

} // namespace nntile::graph
