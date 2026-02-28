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

    //! Build forward (alias for forward). Each Gelu appears as one OpNode.
    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& input);

    // Module forward API
    bool has_custom_backward() const override { return true; }
    void build_backward(const graph::NNGraph::OpNode* op) override;
    graph::NNGraph::TensorNode& forward_impl(
        graph::NNGraph::TensorNode& input) override;
    std::vector<graph::NNGraph::TensorNode*> backward_inputs() const override;

    //! Get string representation
    std::string repr() const override;
};

} // namespace nntile::module
