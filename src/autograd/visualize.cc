/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/autograd/visualize.cc
 * Graph visualization for autograd
 *
 * @version 1.1.0
 * */

#include "nntile/autograd/visualize.hh"
#include <sstream>
#include <iomanip>

namespace nntile::autograd
{

std::string GraphVisualizer::get_node_name(const AutogradTensor& tensor)
{
    return "node_" + std::to_string(node_counter_++);
}

void GraphVisualizer::add_node(const AutogradTensor& tensor, const std::string& name)
{
    NodeInfo info;
    info.name = name;
    info.is_input = !tensor.grad_fn();
    info.requires_grad = tensor.requires_grad();
    if (tensor.grad_fn()) {
        info.op_name = tensor.grad_fn()->name();
    }
    nodes_[tensor.tensor().get()] = info;
}

void GraphVisualizer::build_graph(const AutogradTensor& tensor)
{
    // Skip if already visited
    if (visited_.count(tensor.tensor().get())) {
        return;
    }
    visited_.insert(tensor.tensor().get());

    // Add current node
    std::string name = get_node_name(tensor);
    add_node(tensor, name);

    // Process inputs if this is an operation
    if (tensor.grad_fn()) {
        auto& node = nodes_[tensor.tensor().get()];
        // Get inputs from the function
        const auto& inputs = tensor.grad_fn()->inputs();
        for (const auto& input : inputs) {
            // Recursively build graph for each input
            build_graph(input);
            // Add input to current node's inputs
            node.inputs.push_back(get_node_name(input));
        }
    }
}

std::string GraphVisualizer::node_to_string(const NodeInfo& node) const
{
    std::stringstream ss;
    ss << std::setw(10) << node.name << " [";
    if (node.is_input) {
        ss << "input";
    } else {
        ss << node.op_name;
    }
    ss << ", grad=" << (node.requires_grad ? "true" : "false") << "]";
    return ss.str();
}

std::string GraphVisualizer::visualize(const AutogradTensor& tensor)
{
    // Clear previous state
    nodes_.clear();
    visited_.clear();
    node_counter_ = 0;

    // Build graph
    build_graph(tensor);

    // Generate visualization
    std::stringstream ss;
    ss << "Computation Graph:\n";
    ss << "================\n\n";

    // Print nodes
    for (const auto& [ptr, node] : nodes_) {
        ss << node_to_string(node) << "\n";
        if (!node.inputs.empty()) {
            ss << "  Inputs: ";
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << node.inputs[i];
            }
            ss << "\n";
        }
        ss << "\n";
    }

    return ss.str();
}

} // namespace nntile::autograd
