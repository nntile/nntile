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

#include "nntile/graph/nn/nn_graph.hh"
#include "nntile/graph/nn/op_node.hh"
#include "nntile/graph/nn/tensor_node.hh"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "nntile/graph/tensor/clear.hh"

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

    TensorGraph::DataNode* data =
        tensor_graph_.data(std::move(shape), name, dtype);
    auto node = std::make_unique<TensorNode>(this, data, requires_grad);
    TensorNode* node_ptr = node.get();

    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return node_ptr;
}

NNGraph::TensorNode* NNGraph::tensor(TensorGraph::DataNode* data,
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

    TensorGraph::DataNode* grad_data = tensor_graph_.data(
        tensor->shape(),
        grad_name,
        tensor->dtype());
    auto grad_node = std::make_unique<TensorNode>(this, grad_data, false);
    TensorNode* grad_ptr = grad_node.get();
    tensors_.push_back(std::move(grad_node));
    tensor_by_name_[grad_name] = grad_ptr;

    // Clear freshly registered gradient tensor
    clear(grad_data);

    tensor->set_grad(grad_ptr);
    tensor->set_requires_grad(true);
    return grad_ptr;
}

NNGraph::OpNode* NNGraph::create_op(std::shared_ptr<OpNode> op)
{
    OpNode* ptr = op.get();
    op_nodes_.push_back(std::move(op));
    return ptr;
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

} // namespace nntile::graph
