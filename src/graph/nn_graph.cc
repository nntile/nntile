/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn_graph.cc
 * Implementation of NNGraph class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/nn_graph.hh"

// Include standard headers
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nntile::graph
{

NNGraphTensorNode::NNGraphTensorNode(
    LogicalGraph::TensorNode* data,
    bool requires_grad)
    : data_(data)
    , requires_grad_(requires_grad)
{
    if(data_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraphTensorNode: data tensor is nullptr");
    }
}

std::string NNGraphTensorNode::to_string() const
{
    std::stringstream ss;
    ss << "NNGraphTensorNode(name='" << name() << "', requires_grad="
       << (requires_grad_ ? "true" : "false");
    if(grad_ != nullptr)
    {
        ss << ", grad='" << grad_->name() << "'";
    }
    else
    {
        ss << ", grad=null";
    }
    ss << ", " << spec().to_string() << ")";
    return ss.str();
}

NNGraph::NNGraph(const std::string& name)
    : name_(name)
    , logical_(name)
{
}

NNGraphTensorNode& NNGraph::tensor(
    const TensorSpec& spec,
    const std::string& name,
    bool requires_grad)
{
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("NNGraph::tensor: tensor '" + name +
            "' already exists");
    }

    LogicalGraph::TensorNode& data = logical_.tensor(spec, name);
    auto node = std::make_unique<NNGraphTensorNode>(&data, requires_grad);
    NNGraphTensorNode* node_ptr = node.get();

    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}

void NNGraph::add_op(
    OpType type,
    OpAttrs attrs,
    const std::vector<NNGraphTensorNode*>& inputs,
    const std::vector<NNGraphTensorNode*>& outputs)
{
    std::vector<LogicalGraph::TensorNode*> input_nodes;
    std::vector<LogicalGraph::TensorNode*> output_nodes;
    input_nodes.reserve(inputs.size());
    output_nodes.reserve(outputs.size());

    for(auto* input : inputs)
    {
        if(input == nullptr)
        {
            throw std::invalid_argument(
                "NNGraph::add_op: input tensor is nullptr");
        }
        input_nodes.push_back(input->data_ptr());
    }
    for(auto* output : outputs)
    {
        if(output == nullptr)
        {
            throw std::invalid_argument(
                "NNGraph::add_op: output tensor is nullptr");
        }
        output_nodes.push_back(output->data_ptr());
    }

    logical_.add_op(type, std::move(attrs), input_nodes, output_nodes);
}

void NNGraph::add_op(
    OpType type,
    OpAttrs attrs,
    const std::vector<LogicalGraph::TensorNode*>& inputs,
    const std::vector<LogicalGraph::TensorNode*>& outputs)
{
    logical_.add_op(type, std::move(attrs), inputs, outputs);
}

bool NNGraph::can_remove_tensor(const NNGraphTensorNode* tensor) const
{
    if(tensor == nullptr)
    {
        return false;
    }

    if(tensor->grad() != nullptr)
    {
        if(!logical_.can_remove_tensor(tensor->grad()))
        {
            return false;
        }
    }

    return logical_.can_remove_tensor(tensor->data_ptr());
}

void NNGraph::remove_tensor(NNGraphTensorNode* tensor)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::remove_tensor: tensor is nullptr");
    }

    if(!can_remove_tensor(tensor))
    {
        throw std::runtime_error(
            "NNGraph::remove_tensor: cannot remove tensor '" +
            tensor->name() + "' - it is used by operations");
    }

    const std::string tensor_name = tensor->name();
    LogicalGraph::TensorNode* grad = tensor->grad();
    LogicalGraph::TensorNode* data = tensor->data_ptr();

    if(grad != nullptr)
    {
        logical_.remove_tensor(grad);
    }
    logical_.remove_tensor(data);

    tensor_by_name_.erase(tensor_name);
    auto it = std::find_if(tensors_.begin(), tensors_.end(),
        [tensor](const std::unique_ptr<NNGraphTensorNode>& ptr) {
            return ptr.get() == tensor;
        });
    if(it != tensors_.end())
    {
        tensors_.erase(it);
    }
}

void NNGraph::remove_tensor(const std::string& name)
{
    NNGraphTensorNode* tensor = get_tensor(name);
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::remove_tensor: tensor '" + name + "' not found");
    }
    remove_tensor(tensor);
}

NNGraphTensorNode* NNGraph::get_tensor(const std::string& name)
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

const NNGraphTensorNode* NNGraph::get_tensor(const std::string& name) const
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

std::vector<std::string> NNGraph::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensor_by_name_.size());
    for(const auto& pair : tensor_by_name_)
    {
        names.push_back(pair.first);
    }
    return names;
}

bool NNGraph::requires_grad(const NNGraphTensorNode& tensor) const
{
    return tensor.requires_grad() || tensor.grad() != nullptr;
}

void NNGraph::set_requires_grad(NNGraphTensorNode& tensor, bool requires)
{
    tensor.set_requires_grad(requires);
}

LogicalGraph::TensorNode& NNGraph::get_or_create_grad(
    NNGraphTensorNode& tensor,
    const std::string& grad_name)
{
    if(tensor.grad() != nullptr)
    {
        return *tensor.grad();
    }

    LogicalGraph::TensorNode& grad_tensor = logical_.tensor(
        tensor.spec(),
        grad_name);
    tensor.set_grad(&grad_tensor);
    tensor.set_requires_grad(true);
    return grad_tensor;
}

std::string NNGraph::to_string() const
{
    std::stringstream ss;
    ss << "NNGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", ops=" << num_ops() << ")\n";

    ss << "Tensors:\n";
    for(const auto& t : tensors_)
    {
        ss << "  " << t->to_string() << "\n";
    }

    ss << "Operations:\n";
    for(const auto& op : logical_.ops())
    {
        ss << "  " << op->to_string() << "\n";
    }

    return ss.str();
}

} // namespace nntile::graph
