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
#include <algorithm>
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
LogicalGraphTensorNode& LogicalGraph::tensor(
    const TensorSpec& spec,
    const std::string& name)
{
    // Check name doesn't already exist
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("LogicalGraph::tensor: tensor '" + name +
                "' already exists");
    }

    // Create LogicalGraphTensorNode with unique ID
    auto node = std::make_unique<LogicalGraphTensorNode>(
        next_tensor_id_,
        name,
        spec,
        this
    );
    ++next_tensor_id_;
    LogicalGraphTensorNode* node_ptr = node.get();

    // Store in containers
    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}

//! Add an operation to the graph with specified output tensors
void LogicalGraph::add_op(
    OpType type,
    OpAttrs attrs,
    const std::vector<LogicalGraphTensorNode*>& inputs,
    const std::vector<LogicalGraphTensorNode*>& outputs)
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
LogicalGraphTensorNode* LogicalGraph::get_tensor(const std::string& name)
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get tensor by name (returns nullptr if not found)
const LogicalGraphTensorNode* LogicalGraph::get_tensor(
    const std::string& name) const
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

//! Check if an operation can be removed
bool LogicalGraph::can_remove_op(const OpNode* op) const
{
    if(op == nullptr)
    {
        return false;
    }

    // Check that op belongs to this graph
    if(&op->graph() != this)
    {
        return false;
    }

    // An operation can be removed if none of its output tensors have
    // consumers (other operations that use them)
    for(const auto* output : op->outputs())
    {
        if(!output->consumers().empty())
        {
            return false;
        }
    }

    return true;
}

//! Remove an operation from the graph
void LogicalGraph::remove_op(OpNode* op)
{
    if(op == nullptr)
    {
        throw std::invalid_argument(
            "LogicalGraph::remove_op: op is nullptr");
    }

    // Verify op belongs to this graph
    if(&op->graph() != this)
    {
        throw std::invalid_argument(
            "LogicalGraph::remove_op: operation does not belong to this graph");
    }

    // Check if operation can be removed
    if(!can_remove_op(op))
    {
        throw std::runtime_error(
            "LogicalGraph::remove_op: cannot remove operation - "
            "other operations depend on its outputs");
    }

    // Disconnect from input tensors (remove this op from their consumers)
    for(auto* input : op->inputs())
    {
        input->remove_consumer(op);
    }

    // Disconnect from output tensors (clear their producer)
    for(auto* output : op->outputs())
    {
        output->clear_producer();
    }

    // Find and remove the operation from ops_ vector
    auto it = std::find_if(ops_.begin(), ops_.end(),
        [op](const std::unique_ptr<OpNode>& ptr) {
            return ptr.get() == op;
        });

    if(it != ops_.end())
    {
        ops_.erase(it);
    }
}

//! Check if a tensor can be removed
bool LogicalGraph::can_remove_tensor(const LogicalGraphTensorNode* tensor) const
{
    if(tensor == nullptr)
    {
        return false;
    }

    // Check that tensor belongs to this graph
    if(&tensor->graph() != this)
    {
        return false;
    }

    // A tensor can be removed if:
    // 1. It has no producer (not an output of any operation)
    // 2. It has no consumers (not an input to any operation)
    return !tensor->has_producer() && tensor->consumers().empty();
}

//! Remove a tensor from the graph
void LogicalGraph::remove_tensor(LogicalGraphTensorNode* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "LogicalGraph::remove_tensor: tensor is nullptr");
    }

    // Verify tensor belongs to this graph
    if(&tensor->graph() != this)
    {
        throw std::invalid_argument(
            "LogicalGraph::remove_tensor: tensor does not belong to this graph");
    }

    // Check if tensor can be removed
    if(!can_remove_tensor(tensor))
    {
        std::string reason;
        if(tensor->has_producer())
        {
            reason = "it is produced by an operation";
        }
        else if(!tensor->consumers().empty())
        {
            reason = "it is consumed by " +
                std::to_string(tensor->consumers().size()) + " operation(s)";
        }
        throw std::runtime_error(
            "LogicalGraph::remove_tensor: cannot remove tensor '" +
            tensor->name() + "' - " + reason);
    }

    // Remove from name map
    tensor_by_name_.erase(tensor->name());

    // Find and remove from tensors_ vector
    auto it = std::find_if(tensors_.begin(), tensors_.end(),
        [tensor](const std::unique_ptr<LogicalGraphTensorNode>& ptr) {
            return ptr.get() == tensor;
        });

    if(it != tensors_.end())
    {
        tensors_.erase(it);
    }
}

//! Remove a tensor by name
void LogicalGraph::remove_tensor(const std::string& name)
{
    LogicalGraphTensorNode* tensor = get_tensor(name);
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "LogicalGraph::remove_tensor: tensor '" + name + "' not found");
    }
    remove_tensor(tensor);
}

} // namespace nntile::graph
