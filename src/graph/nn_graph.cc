/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph.cc
 * Implementation of NNGraph class.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/nn_graph.hh"
#include "nntile/graph/nn_graph/op_node.hh"
#include "nntile/graph/nn_graph/tensor_node.hh"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "nntile/graph/logical_graph_ops.hh"

namespace nntile::graph
{

NNGraph::~NNGraph() = default;

NNGraph::NNGraph(const std::string& name)
    : name_(name)
    , logical_(name)
{
}

NNGraph::TensorNode* NNGraph::tensor(
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
    auto node = std::make_unique<TensorNode>(this, &data, requires_grad);
    TensorNode* node_ptr = node.get();

    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return node_ptr;
}

NNGraph::TensorNode* NNGraph::tensor(LogicalGraph::TensorNode& data,
                                     bool requires_grad)
{
    if(&data.graph() != &logical_)
    {
        throw std::invalid_argument(
            "NNGraph::tensor: tensor must belong to this graph's logical graph");
    }
    auto node = std::make_unique<TensorNode>(this, &data, requires_grad);
    TensorNode* node_ptr = node.get();
    tensors_.push_back(std::move(node));
    tensor_by_name_[data.name()] = node_ptr;
    return node_ptr;
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

    // Propagate grad_required from inputs to outputs (PyTorch-style)
    bool any_input_requires_grad = false;
    for(const auto* in : inputs)
    {
        if(in && in->requires_grad())
        {
            any_input_requires_grad = true;
            break;
        }
    }
    for(auto* out : outputs)
    {
        if(out)
        {
            out->set_requires_grad(any_input_requires_grad);
        }
    }
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

bool NNGraph::requires_grad(const TensorNode* tensor) const
{
    return tensor != nullptr &&
           (tensor->requires_grad() || tensor->grad() != nullptr);
}

void NNGraph::set_requires_grad(TensorNode* tensor, bool requires)
{
    if(tensor != nullptr)
    {
        tensor->set_requires_grad(requires);
    }
}

bool NNGraph::is_first_grad(const TensorNode* tensor) const
{
    return tensor != nullptr && tensor->grad() == nullptr;
}

NNGraph::TensorNode* NNGraph::get_or_create_grad(
    TensorNode* tensor,
    const std::string& grad_name)
{
    if(tensor == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::get_or_create_grad: tensor is nullptr");
    }
    if(tensor->grad() != nullptr)
    {
        return tensor->grad();
    }

    LogicalGraph::TensorNode& grad_tensor = logical_.tensor(
        tensor->shape(),
        grad_name,
        tensor->dtype());
    auto grad_node = std::make_unique<TensorNode>(this, &grad_tensor, false);
    TensorNode* grad_ptr = grad_node.get();
    tensors_.push_back(std::move(grad_node));
    tensor_by_name_[grad_name] = grad_ptr;

    // Clear freshly registered gradient tensor
    clear(grad_tensor);

    tensor->set_grad(grad_ptr);
    tensor->set_requires_grad(true);
    return grad_ptr;
}

NNGraph::OpNode* NNGraph::create_op(
    std::vector<TensorNode*> inputs,
    std::vector<TensorNode*> outputs,
    std::shared_ptr<void> attrs,
    std::function<void(const OpNode*)> backward_fn,
    std::vector<TensorNode*> buffers)
{
    auto op = std::make_unique<OpNode>(
        std::move(inputs), std::move(outputs), std::move(attrs),
        std::move(backward_fn), std::move(buffers));
    OpNode* ptr = op.get();
    op_nodes_.push_back(std::move(op));
    return ptr;
}

void NNGraph::wrap_with_module_op(
    std::vector<TensorNode*> inputs,
    std::vector<TensorNode*> outputs,
    std::function<void(const OpNode*)> backward_fn)
{
    OpNode* op = create_op(
        std::move(inputs),
        std::move(outputs),
        nullptr,
        std::move(backward_fn));
    for(TensorNode* out : op->outputs())
    {
        if(out != nullptr)
        {
            out->set_producer(op);
        }
    }
}

void NNGraph::wrap_with_module_op(
    std::vector<TensorNode*> inputs,
    TensorNode* output,
    std::function<void(const OpNode*)> backward_fn)
{
    if(output == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::wrap_with_module_op: output is nullptr");
    }
    wrap_with_module_op(
        std::move(inputs),
        std::vector<TensorNode*>{output},
        std::move(backward_fn));
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
