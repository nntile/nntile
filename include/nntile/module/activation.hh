/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/activation.hh
 * Configurable activation module (gelu, gelutanh, relu, silu).
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <stdexcept>
#include <string>

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Activation function type
enum class ActivationType
{
    GELU,
    GELUTANH,
    RELU,
    SILU
};

//! Convert activation type to string
inline const char* activation_type_to_string(ActivationType type)
{
    switch(type)
    {
        case ActivationType::GELU: return "gelu";
        case ActivationType::GELUTANH: return "gelutanh";
        case ActivationType::RELU: return "relu";
        case ActivationType::SILU: return "silu";
    }
    return "unknown";
}

//! Parse activation type from string
inline ActivationType activation_type_from_string(const std::string& s)
{
    if(s == "gelu") return ActivationType::GELU;
    if(s == "gelutanh") return ActivationType::GELUTANH;
    if(s == "relu") return ActivationType::RELU;
    if(s == "silu") return ActivationType::SILU;
    throw std::invalid_argument("Unknown activation type: " + s);
}

//! Configurable activation module using graph API
//! Supports: gelu, gelutanh, relu, silu
class Activation : public Module
{
private:
    ActivationType type_;
    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name (used to generate unique tensor names)
    //! @param type Activation function type
    Activation(graph::NNGraph& graph,
               const std::string& name,
               ActivationType type = ActivationType::GELU);

    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    //! Forward: calls build_forward
    graph::NNGraph::TensorNode& operator()(graph::NNGraph::TensorNode& input)
    {
        return build_forward(input);
    }

    //! Get string representation
    std::string repr() const override;

    //! Get activation type
    ActivationType type() const { return type_; }
};

} // namespace nntile::module
