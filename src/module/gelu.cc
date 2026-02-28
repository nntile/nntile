/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/gelu.cc
 * GeLU module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/gelu.hh"

// Include standard headers
#include <stdexcept>

namespace nntile::module
{

Gelu::Gelu(graph::NNGraph& graph, const std::string& name)
    : Module(graph, name)
{
}

graph::NNGraph::TensorNode& Gelu::build_forward(
    graph::NNGraph::TensorNode& input)
{
    return (*this)(input);
}

graph::NNGraph::TensorNode& Gelu::operator()(
    graph::NNGraph::TensorNode& input)
{
    input_tensor_ = &input;
    output_tensor_ = graph::gelu(&input, tensor_name("output"));
    return *output_tensor_;
}

std::string Gelu::repr() const
{
    return "Gelu()";
}

} // namespace nntile::module
