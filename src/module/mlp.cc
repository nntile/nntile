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
Mlp::Mlp(const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         Index output_dim,
         graph::DataType dtype)
    : Module(name)
    , fc1_(name + "_fc1", input_dim, intermediate_dim, dtype)
    , fc2_(name + "_fc2", intermediate_dim, output_dim, dtype)
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
Mlp::Mlp(const std::string& name,
         Index input_dim,
         Index intermediate_dim,
         graph::DataType dtype)
    : Mlp(name, input_dim, intermediate_dim, input_dim, dtype)
{
}

//! Build forward operations
graph::TensorNode& Mlp::build_forward(
    graph::LogicalGraph& graph,
    graph::TensorNode& input)
{
    // Store input reference
    input_tensor_ = &input;

    // fc1: input -> hidden
    hidden_tensor_ = &fc1_.build_forward(graph, input);

    // GELU activation: hidden -> activation
    // Create activation output tensor
    activation_tensor_ = &graph.tensor(
        hidden_tensor_->spec(),
        tensor_name("activation"));

    graph.add_op(
        graph::OpType::GELU,
        graph::OpAttrs{graph::GeluAttrs{}},
        {hidden_tensor_},
        {activation_tensor_}
    );

    // fc2: activation -> output
    output_tensor_ = &fc2_.build_forward(graph, *activation_tensor_);

    forward_built_ = true;
    return *output_tensor_;
}

//! Build backward operations using gradient registry
void Mlp::build_backward(
    graph::LogicalGraph& graph,
    graph::GradientRegistry& grad_reg)
{
    // Check that forward has been built
    if(!forward_built_)
    {
        throw std::runtime_error(
            "Mlp::build_backward: forward not built - "
            "call build_forward first");
    }

    // Backward through fc2
    fc2_.build_backward(graph, grad_reg);

    // Get gradient of activation tensor (output of GELU, input of fc2)
    graph::TensorNode* grad_activation = grad_reg.get_grad(*activation_tensor_);
    if(!grad_activation)
    {
        throw std::runtime_error(
            "Mlp::build_backward: no gradient for activation tensor");
    }

    // GELU backward: compute gradient of hidden tensor
    bool first_hidden_grad = grad_reg.is_first_grad(*hidden_tensor_);
    graph::TensorNode& grad_hidden = grad_reg.get_or_create_grad(
        graph, *hidden_tensor_, tensor_name("hidden_grad"));

    // gelu_backward computes: grad_hidden = grad_activation * gelu'(hidden)
    // Note: GELU_BACKWARD takes (input, grad_output) and produces grad_input
    if(first_hidden_grad)
    {
        graph.add_op(
            graph::OpType::GELU_BACKWARD,
            graph::OpAttrs{graph::GeluBackwardAttrs{}},
            {hidden_tensor_, grad_activation},
            {&grad_hidden}
        );
    }
    else
    {
        // Need to accumulate - create temp and add
        // For simplicity, assume first contribution for now
        // TODO: handle accumulation properly
        graph.add_op(
            graph::OpType::GELU_BACKWARD,
            graph::OpAttrs{graph::GeluBackwardAttrs{}},
            {hidden_tensor_, grad_activation},
            {&grad_hidden}
        );
    }

    // Register gradient for hidden tensor so fc1 can find it
    // (fc1's output is hidden_tensor_)
    // Note: fc1_.output_tensor() == hidden_tensor_, so grad_reg already has it

    // Backward through fc1
    fc1_.build_backward(graph, grad_reg);
}

//! Get string representation with dimensions
std::string Mlp::repr() const
{
    return "Mlp(in=" + std::to_string(input_dim_) +
           ", intermediate=" + std::to_string(intermediate_dim_) +
           ", out=" + std::to_string(output_dim_) + ")";
}

} // namespace nntile::module