/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical_graph.cc
 * Implementation of LogicalGraph class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical_graph.hh"

// Include standard headers
#include <sstream>
#include <stdexcept>

// Include third-party headers

// Include other NNTile headers

namespace nntile::graph
{

LogicalGraph::LogicalGraph(const std::string& name)
    : name_(name)
{
}

//! Create an input tensor (not produced by any operation)
TensorNode& LogicalGraph::tensor(
    const TensorSpec& spec,
    const std::string& name)
{
    // Check name doesn't already exist
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("LogicalGraph::tensor: tensor '" + name +
                "' already exists");
    }

    // Create TensorNode with unique ID
    auto node = std::make_unique<TensorNode>(
        next_tensor_id_,
        name,
        spec,
        this
    );
    ++next_tensor_id_;
    TensorNode* node_ptr = node.get();

    // Store in containers
    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}

//! Add an operation to the graph with specified output tensors
void LogicalGraph::add_op(
    OpType type,
    OpAttrs attrs,
    const std::vector<TensorNode*>& inputs,
    const std::vector<TensorNode*>& outputs)
{
    // Validate all inputs belong to this graph
    for(const auto* input : inputs)
    {
        if(&input->graph() != this)
        {
            throw std::invalid_argument(
                "LogicalGraph::add_op: input tensor '" + input->name() +
                "' does not belong to this graph");
        }
    }

    // Validate all outputs belong to this graph
    for(const auto* output : outputs)
    {
        if(&output->graph() != this)
        {
            throw std::invalid_argument(
                "LogicalGraph::add_op: output tensor '" + output->name() +
                "' does not belong to this graph");
        }
    }

    // Create OpNode
    auto op = std::make_unique<OpNode>(
        next_op_id_,
        type,
        attrs,
        this
    );
    ++next_op_id_;
    OpNode* op_ptr = op.get();

    // Wire up inputs
    for(auto* input : inputs)
    {
        op_ptr->add_input(input);
    }

    // Wire up outputs
    for(auto* output : outputs)
    {
        op_ptr->add_output(output);
    }

    // Store operation
    ops_.push_back(std::move(op));
}

//! Get tensor by name (returns nullptr if not found)
TensorNode* LogicalGraph::get_tensor(const std::string& name)
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get tensor by name (returns nullptr if not found)
const TensorNode* LogicalGraph::get_tensor(const std::string& name) const
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get all tensor names
std::vector<std::string> LogicalGraph::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensor_by_name_.size());
    for(const auto& pair : tensor_by_name_)
    {
        names.push_back(pair.first);
    }
    return names;
}

//! String representation
std::string LogicalGraph::to_string() const
{
    std::stringstream ss;
    ss << "LogicalGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", ops=" << num_ops() << ")\n";

    ss << "Tensors:\n";
    for(const auto& t : tensors_)
    {
        ss << "  " << t->to_string() << "\n";
    }

    ss << "Operations:\n";
    for(const auto& op : ops_)
    {
        ss << "  " << op->to_string() << "\n";
    }

    return ss.str();
}

} // namespace nntile::graph
