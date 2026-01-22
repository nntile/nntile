/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/mlp.cc
 * MLP module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/mlp.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::module
{

//! Constructor: creates MLP with specified dimensions
Mlp::Mlp(graph::NNGraph& graph,
         const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         Index output_dim,
         graph::DataType dtype)
    : Module(graph, name)
    , fc1_(graph, name + "_fc1", input_dim, intermediate_dim, dtype)
    , fc2_(graph, name + "_fc2", intermediate_dim, output_dim, dtype)
    , input_dim_(input_dim)
    , intermediate_dim_(intermediate_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Register submodules
    register_module("fc1", &fc1_);
    register_module("fc2", &fc2_);
}

//! Constructor: creates MLP where output_dim == input_dim
Mlp::Mlp(graph::NNGraph& graph,
         const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         graph::DataType dtype)
    : Mlp(graph, name, input_dim, intermediate_dim, input_dim, dtype)
{
}

//! Build forward operations
graph::NNGraphTensorNode& Mlp::build_forward(graph::NNGraphTensorNode& input)
{
    // Store input reference
    input_tensor_ = &input;

    // fc1: input -> hidden
    hidden_tensor_ = &fc1_.build_forward(input);

    // GELU activation: hidden -> activation
    // Create activation output tensor
    activation_tensor_ = &graph_.tensor(
        hidden_tensor_->shape(),
        tensor_name("activation"),
        hidden_tensor_->dtype(),
        graph_.requires_grad(*hidden_tensor_));

    graph_.add_op(
        graph::OpType::GELU,
        graph::OpAttrs{graph::GeluAttrs{}},
        {hidden_tensor_},
        {activation_tensor_}
    );

    // Mark that activation tensor needs gradient (for GELU backward)
    // This is internal to the module

    // fc2: activation -> output
    output_tensor_ = &fc2_.build_forward(*activation_tensor_);

    forward_built_ = true;
    return *output_tensor_;
}

//! Build backward operations using gradient tracking
void Mlp::build_backward()
{
    // Check that forward has been built
    if(!forward_built_)
    {
        throw std::runtime_error(
            "Mlp::build_backward: forward not built - "
            "call build_forward first");
    }

    // Mark that activation tensor needs gradient (internal requirement)
    graph_.set_requires_grad(*activation_tensor_, true);

    // Backward through fc2
    fc2_.build_backward();

    // Get gradient of activation tensor (output of GELU, input of fc2)
    graph::LogicalGraph::TensorNode* grad_activation =
        activation_tensor_->grad();
    if(!grad_activation)
    {
        throw std::runtime_error(
            "Mlp::build_backward: no gradient for activation tensor");
    }

    // GELU backward: compute gradient of hidden tensor
    // Mark that hidden tensor needs gradient (for fc1 backward)
    graph_.set_requires_grad(*hidden_tensor_, true);

    graph::LogicalGraph::TensorNode& grad_hidden = graph_.get_or_create_grad(
        *hidden_tensor_, tensor_name("hidden_grad"));

    // gelu_backward computes: grad_hidden = grad_activation * gelu'(hidden)
    // Note: GELU_BACKWARD takes (input, grad_output) and produces grad_input
    graph_.add_op(
        graph::OpType::GELU_BACKWARD,
        graph::OpAttrs{graph::GeluBackwardAttrs{}},
        {hidden_tensor_->data_ptr(), grad_activation, &grad_hidden},
        {&grad_hidden}
    );

    // Propagate requires_grad to fc1's input if MLP's input requires grad
    if(graph_.requires_grad(*input_tensor_))
    {
        graph_.set_requires_grad(*fc1_.input_tensor(), true);
    }

    // Backward through fc1
    fc1_.build_backward();
}

//! Get string representation with dimensions
std::string Mlp::repr() const
{
    return "Mlp(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) + ")";
}

} // namespace nntile::module
