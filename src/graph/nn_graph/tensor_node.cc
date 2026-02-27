/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/nn_graph/tensor_node.cc
 * NNGraph::TensorNode implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/nn_graph/tensor_node.hh"
#include "nntile/graph/nn_graph/op_node.hh"

#include <deque>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nntile::graph
{

NNGraph::TensorNode::TensorNode(
    LogicalGraph::TensorNode* data,
    bool requires_grad
)
    : graph_(nullptr)
    , data_(data)
    , requires_grad_(requires_grad)
{
    if(data_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::TensorNode: data tensor is nullptr");
    }
}

NNGraph::TensorNode::TensorNode(
    NNGraph* graph,
    LogicalGraph::TensorNode* data,
    bool requires_grad
)
    : graph_(graph)
    , data_(data)
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

NNGraph& NNGraph::TensorNode::graph()
{
    if(graph_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::TensorNode::graph: tensor has no graph reference");
    }
    return *graph_;
}

void NNGraph::TensorNode::set_producer(OpNode* op)
{
    producer_ = op;
}

void NNGraph::TensorNode::backward()
{
    if(graph_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::TensorNode::backward: tensor has no graph reference");
    }

    // Build reverse topological order using producer->inputs
    std::deque<TensorNode*> rev_topo;
    std::set<TensorNode*> visited;
    std::deque<TensorNode*> stack = {this};

    while(!stack.empty())
    {
        TensorNode* t = stack.back();
        stack.pop_back();
        if(visited.count(t))
        {
            continue;
        }
        visited.insert(t);
        rev_topo.push_back(t);

        if(t->producer() != nullptr)
        {
            for(TensorNode* in : t->producer()->inputs())
            {
                if(in != nullptr && in->requires_grad() && visited.count(in) == 0)
                {
                    stack.push_back(in);
                }
            }
        }
    }

    if(grad_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::TensorNode::backward: grad must be set before backward(). "
            "Use get_or_create_grad() and fill/bind the gradient.");
    }

    // Call each tensor's producer->backward (adds gradient LogicalGraph ops)
    for(TensorNode* t : rev_topo)
    {
        if(t->producer() == nullptr)
        {
            continue;
        }

        if(t->grad() == nullptr)
        {
            continue;
        }

        t->producer()->run_backward();
    }
}

} // namespace nntile::graph
