/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/mse_loss.cc
 * MSE loss module implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/module/mse_loss.hh"

#include <stdexcept>

namespace nntile::module
{

MseLoss::MseLoss(graph::NNGraph& graph,
                 const std::string& name,
                 graph::DataType dtype)
    : ModuleBase(graph, name)
    , dtype_(dtype)
{
}

graph::NNGraph::TensorNode& MseLoss::build_forward(
    graph::NNGraph::TensorNode& input)
{
    input_tensor_ = &input;

    // y = norm(x)
    graph::LogicalGraph::TensorNode& norm_data =
        graph_.logical_graph().tensor({}, tensor_name("norm"), dtype_);
    graph::clear(norm_data);
    graph::norm(input.data(), norm_data, 1.0, 0.0);

    // loss = y * y
    graph::LogicalGraph::TensorNode& loss_data =
        graph::multiply(norm_data, norm_data, tensor_name("loss"));

    loss_tensor_ = graph_.tensor(loss_data);
    return *loss_tensor_;
}

std::string MseLoss::repr() const
{
    return "MseLoss()";
}

} // namespace nntile::module
