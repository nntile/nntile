/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/nn/graph.cc
 * Implementation of NNGraph class (include/nntile/graph/nn/graph.hh).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn/graph.hh"
#include "nntile/graph/nn/graph_op_node.hh"
#include "nntile/graph/nn/graph_data_node.hh"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace nntile::graph
{

NNGraph::~NNGraph() = default;

NNGraph::NNGraph(const std::string& name)
    : name_(name)
    , tensor_graph_(name)
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

    TensorGraph::TensorNode* data =
        tensor_graph_.data(std::move(shape), name, dtype);
    auto node = std::make_unique<TensorNode>(this, data, requires_grad);
    TensorNode* node_ptr = node.get();

    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return node_ptr;
}

NNGraph::TensorNode* NNGraph::tensor(TensorGraph::TensorNode* data,
                                     bool requires_grad)
{
    if(data == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::tensor: data tensor must be non-null");
    }
    if(data->graph() != &tensor_graph_)
    {
        throw std::invalid_argument(
            "NNGraph::tensor: tensor must belong to this graph's tensor graph");
    }
    if(tensor_by_name_.count(data->name()) > 0)
    {
        throw std::invalid_argument("NNGraph::tensor: tensor '" + data->name() +
            "' already exists");
    }
    auto node = std::make_unique<TensorNode>(this, data, requires_grad);
    TensorNode* node_ptr = node.get();
    tensors_.push_back(std::move(node));
    tensor_by_name_[data->name()] = node_ptr;
    return node_ptr;
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

std::pair<NNGraph::TensorNode*, bool> NNGraph::get_or_create_grad(
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
        if(tensor->grad()->name() != grad_name)
        {
            throw std::invalid_argument(
                "NNGraph::get_or_create_grad: tensor '" + tensor->name() +
                "' already has gradient '" + tensor->grad()->name() +
                "' but caller requested '" + grad_name + "'");
        }
        return {tensor->grad(), false};
    }

    TensorGraph::TensorNode* grad_data = tensor_graph_.data(
        tensor->shape(),
        grad_name,
        tensor->dtype());
    auto grad_node = std::make_unique<TensorNode>(this, grad_data, false);
    TensorNode* grad_ptr = grad_node.get();
    tensors_.push_back(std::move(grad_node));
    tensor_by_name_[grad_name] = grad_ptr;

    tensor->set_grad(grad_ptr);
    tensor->set_requires_grad(true);
    return {grad_ptr, true};
}

void NNGraph::clear_op_nodes()
{
    op_nodes_.clear();
}

void NNGraph::clear_producers_on_tensors()
{
    for(auto& t : tensors_)
    {
        if(t->has_producer())
        {
            t->set_producer(nullptr);
        }
    }
}

NNGraph::NoGradGuard::NoGradGuard(NNGraph& graph)
    : graph_(graph)
    , prev_(graph.grad_enabled_)
{
    graph_.grad_enabled_ = false;
}

NNGraph::NoGradGuard::~NoGradGuard()
{
    graph_.grad_enabled_ = prev_;
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
    for(const auto& op : tensor_graph_.ops())
    {
        ss << "  " << op->op_name() << "(id=" << op->id() << ")\n";
    }

    return ss.str();
}

void NNGraph::register_op(std::shared_ptr<OpNode> op)
{
    const bool need_backward =
        is_grad_enabled() && any_input_requires_grad(op->inputs());

    if(!need_backward)
    {
        return;
    }

    OpNode* op_nn = op.get();
    op_nodes_.push_back(std::move(op));

    for(TensorNode* out : op_nn->outputs())
    {
        if(out != nullptr)
        {
            out->set_producer(op_nn);
        }
    }
}

bool any_input_requires_grad(
    const std::vector<NNGraph::TensorNode*>& inputs)
{
    for(const auto* in : inputs)
    {
        if(in != nullptr && in->requires_grad())
        {
            return true;
        }
    }
    return false;
}

} // namespace nntile::graph
