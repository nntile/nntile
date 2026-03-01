/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph/tensor_node.hh
 * NNGraph::TensorNode - tensor node with autograd.
 *
 * Include this only via nn_graph/nn_graph.hh (after NNGraph is declared).
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/nn_graph/nn_graph.hh>

namespace nntile::graph
{

//! Tensor node in NNGraph. Full definition of nested class NNGraph::TensorNode.
class NNGraph::TensorNode
{
    friend class NNGraph;

private:
    NNGraph* graph_ = nullptr;
    LogicalGraph::TensorNode* data_ = nullptr;
    TensorNode* grad_ = nullptr;
    bool requires_grad_ = true;
    OpNode* producer_ = nullptr;

public:
    TensorNode(LogicalGraph::TensorNode* data, bool requires_grad = true);
    TensorNode(NNGraph* graph, LogicalGraph::TensorNode* data,
               bool requires_grad = true);

    // Accessors for underlying logical nodes
    LogicalGraph::TensorNode& data() { return *data_; }
    const LogicalGraph::TensorNode& data() const { return *data_; }
    LogicalGraph::TensorNode* data_ptr() { return data_; }
    const LogicalGraph::TensorNode* data_ptr() const { return data_; }

    TensorNode* grad() { return grad_; }
    const TensorNode* grad() const { return grad_; }
    bool has_grad() const { return grad_ != nullptr; }

    // Gradient requirement
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires) { requires_grad_ = requires; }

    // Autograd: NNGraph-level producer (not LogicalGraph)
    bool is_leaf() const { return producer_ == nullptr; }
    bool has_producer() const { return producer_ != nullptr; }
    const OpNode* producer() const { return producer_; }

    // Set by NNGraph op that created this tensor (may use multiple LogicalGraph ops)
    void set_producer(OpNode* op);

    // Autograd: propagate upstream gradient through the computation graph.
    //! Grad must be set beforehand (get_or_create_grad + fill/bind).
    //! Does NOT fill grad with ones - user must provide upstream gradient.
    void backward();

    // Graph access (for operations that deduce graph from tensor)
    NNGraph& graph();

    // Convenience accessors (forwarded to data tensor)
    const std::string& name() const { return data_->name(); }
    DataType dtype() const { return data_->dtype(); }
    const std::vector<Index>& shape() const { return data_->shape(); }
    Index ndim() const { return data_->ndim(); }

    // Input/output marking (forwarded to data tensor)
    bool is_input() const { return data_->is_input(); }
    void mark_input(bool is_input = true) { data_->mark_input(is_input); }
    bool is_output() const { return data_->is_output(); }
    void mark_output(bool is_output = true) { data_->mark_output(is_output); }

    // String representation
    std::string to_string() const;

private:
    void set_grad(TensorNode* grad) { grad_ = grad; }
    void set_graph(NNGraph* graph) { graph_ = graph; }
};

} // namespace nntile::graph
