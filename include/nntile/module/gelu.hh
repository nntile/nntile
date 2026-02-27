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
private:
    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

public:
    //! Constructor
    //! @param graph The neural network graph this module belongs to
    //! @param name Module name (used to generate unique tensor names)
    Gelu(graph::NNGraph& graph, const std::string& name);

    //! Build forward operation and return output tensor.
    //! Uses autograd Gelu; backward via tensor.backward().
    //! @param input Input tensor node
    //! @return Reference to the created output tensor
    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    //! Get string representation
    std::string repr() const override;
};

} // namespace nntile::module
