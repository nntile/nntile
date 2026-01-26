/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/gelu.hh
 * GeLU module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! GeLU activation module using graph API
//! Computes: output = gelu(input)
class Gelu : public Module
{
public:
    //! Constructor
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name (used to generate unique tensor names)
    Gelu(graph::NNGraph& graph, const std::string& name);

    //! Build forward operation and return output tensor
    //! @param input Input tensor node
    //! @return Reference to the created output tensor
    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input) override;

    //! Build backward operations using grad fields on NNGraph::TensorNode
    //!
    //! This method:
    //! 1. Looks up gradient of output tensor from output_tensor()->grad()
    //! 2. Computes gradient of input tensor if requires_grad is set
    //!
    //! @throws std::runtime_error if build_forward was not called first
    void build_backward() override;

    //! Get string representation
    std::string repr() const override;
};

} // namespace nntile::module
