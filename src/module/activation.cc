/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/activation.cc
 * Configurable activation module implementation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/activation.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::module
{

Activation::Activation(graph::NNGraph* graph,
                       const std::string& name,
                       ActivationType type)
    : Module(graph, name)
    , type_(type)
{
}

graph::NNGraph::TensorNode* Activation::forward(
    graph::NNGraph::TensorNode* input)
{
    if(input == nullptr)
    {
        throw std::invalid_argument(
            "Activation::forward: input tensor must be non-null");
    }
    input_tensor_ = input;
    switch(type_)
    {
        case ActivationType::GELU:
            output_tensor_ = graph::gelu(input, tensor_name("output"));
            break;
        case ActivationType::GELUTANH:
            output_tensor_ = graph::gelutanh(input, tensor_name("output"));
            break;
        case ActivationType::RELU:
            output_tensor_ = graph::relu(input, tensor_name("output"));
            break;
        case ActivationType::SILU:
            output_tensor_ = graph::silu(input, tensor_name("output"));
            break;
    }
    return output_tensor_;
}

std::string Activation::repr() const
{
    return std::string("Activation(") + activation_type_to_string(type_) + ")";
}

} // namespace nntile::module
