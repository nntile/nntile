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
#include <sstream>
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include "nntile/graph/logical/clear.hh"

namespace nntile::graph
{

NNGraph::TensorNode::TensorNode(
    LogicalGraph::TensorNode* data,
    bool requires_grad
)
    : data_(data)
    , requires_grad_(requires_grad)
{
    if(data_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::TensorNode: data tensor is nullptr");
    }
}

std::string NNGraph::TensorNode::to_string() const
{
    std::stringstream ss;
    ss << "NNGraph::TensorNode(name='" << name() << "', requires_grad="
       << (requires_grad_ ? "true" : "false");
    if(grad_ != nullptr)
    {
        ss << ", grad='" << grad_->name() << "'";
    }
    else
    {
        ss << ", grad=null";
    }
    ss << ", shape=[";
    for(size_t i = 0; i < shape().size(); ++i)
    {
        if(i > 0)
        {
            ss << ", ";
        }
        ss << shape()[i];
    }
    ss << "], dtype=" << dtype_to_string(dtype()) << ")";
    return ss.str();
}

NNGraph::NNGraph(const std::string& name)
    : name_(name)
    , logical_(name)
{
}

NNGraph::TensorNode& NNGraph::tensor(
    std::vector<Index> shape,
    const std::string& name,
    DataType dtype,
    bool requires_grad
)
{
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("NNGraph::tensor: tensor '" + name +
            "' already exists");
    }

    LogicalGraph::TensorNode& data = logical_.tensor(std::move(shape), name, dtype);
    auto node = std::make_unique<TensorNode>(&data, requires_grad);
    TensorNode* node_ptr = node.get();

    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}

void NNGraph::add_op(
    OpType type,
    OpAttrs attrs,
    const std::vector<TensorNode*>& inputs,
    const std::vector<TensorNode*>& outputs,
    const std::string& name)
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

    logical_.add_op(type, std::move(attrs), input_nodes, output_nodes, name);
}

NNGraph::TensorNode* NNGraph::get_tensor(const std::string& name)
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

const NNGraph::TensorNode* NNGraph::get_tensor(const std::string& name) const
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

bool NNGraph::requires_grad(const TensorNode& tensor) const
{
    return tensor.requires_grad() || tensor.grad() != nullptr;
}

void NNGraph::set_requires_grad(TensorNode& tensor, bool requires)
{
    tensor.set_requires_grad(requires);
}

NNGraph::TensorNode& NNGraph::get_or_create_grad(
    TensorNode& tensor,
    const std::string& grad_name)
{
    if(tensor.grad() != nullptr)
    {
        return *tensor.grad();
    }

    LogicalGraph::TensorNode& grad_tensor = logical_.tensor(
        tensor.shape(),
        grad_name,
        tensor.dtype());
    auto grad_node = std::make_unique<TensorNode>(&grad_tensor, false);
    TensorNode* grad_ptr = grad_node.get();
    tensors_.push_back(std::move(grad_node));
    tensor_by_name_[grad_name] = grad_ptr;

    // Clear freshly registered gradient tensor
    clear(grad_tensor);

    tensor.set_grad(grad_ptr);
    tensor.set_requires_grad(true);
    return *grad_ptr;
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
