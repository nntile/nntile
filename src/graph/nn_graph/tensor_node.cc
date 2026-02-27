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
#include <unordered_set>
#include <sstream>
#include <stdexcept>
#include <vector>

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

    // Build reverse topological order: DFS post-order from output toward inputs.
    // Post-order ensures we process a node only after all its consumers (in
    // forward) have contributed to its grad. Handles diamond patterns.
    std::vector<TensorNode*> post_order;
    post_order.reserve(64);
    std::unordered_set<TensorNode*> visited;
    std::unordered_set<TensorNode*> done;
    std::vector<TensorNode*> stack = {this};

    while(!stack.empty())
    {
        TensorNode* t = stack.back();
        stack.pop_back();
        if(done.count(t))
        {
            continue;
        }
        if(visited.count(t))
        {
            post_order.push_back(t);
            done.insert(t);
            continue;
        }
        visited.insert(t);
        stack.push_back(t);  // Re-push to add after children
        if(t->producer() != nullptr)
        {
            for(TensorNode* in : t->producer()->inputs())
            {
                if(in != nullptr && in->requires_grad() && !done.count(in))
                {
                    stack.push_back(in);
                }
            }
        }
        else
        {
            post_order.push_back(t);
            done.insert(t);
        }
    }

    // Reverse for backward: process output first, then toward inputs
    std::deque<TensorNode*> rev_topo(post_order.rbegin(), post_order.rend());

    if(grad_ == nullptr)
    {
        throw std::invalid_argument(
            "NNGraph::TensorNode::backward: grad must be set before backward(). "
            "Use get_or_create_grad() and fill/bind the gradient.");
    }

    // Call each OpNode's backward once (multi-output ops share one producer)
    std::unordered_set<const OpNode*> op_done;
    for(TensorNode* t : rev_topo)
    {
        const OpNode* op = t->producer();
        if(op == nullptr || op_done.count(op))
        {
            continue;
        }
        // Run backward when any output has grad (backward_fn uses op->outputs())
        bool any_has_grad = false;
        for(TensorNode* out : op->outputs())
        {
            if(out != nullptr && out->grad() != nullptr)
            {
                any_has_grad = true;
                break;
            }
        }
        if(!any_has_grad)
        {
            continue;
        }
        op_done.insert(op);
        op->run_backward();
    }
}

} // namespace nntile::graph
