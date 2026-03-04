/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/graph_data_node.hh
 * NNGraph::TensorNode - tensor node with autograd.
 *
 * Include via nn.hh or nn/graph.hh (after graph_decl.hh).
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>
#include <nntile/graph/nn/graph_decl.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Tensor node in NNGraph. Holds data_ (TensorGraph::TensorNode) for ops;
//! adds grad_, requires_grad_, producer_. Shape/dtype/name delegate to data_.
class NNGraph::TensorNode
{
    friend class NNGraph;

private:
    NNGraph* graph_ = nullptr;
    TensorGraph::TensorNode* data_ = nullptr;
    TensorNode* grad_ = nullptr;
    bool requires_grad_ = true;
    OpNode* producer_ = nullptr;

public:
    TensorNode(
        NNGraph* graph,
        TensorGraph::TensorNode* data,
        bool requires_grad = true);

    // Shape/dtype/name delegate to underlying data node
    const std::vector<Index>& shape() const { return data_->shape(); }
    Index ndim() const { return static_cast<Index>(data_->shape().size()); }
    DataType dtype() const { return data_->dtype(); }
    const std::string& name() const { return data_->name(); }

    // Accessors for underlying data node
    TensorGraph::TensorNode* data() { return data_; }
    const TensorGraph::TensorNode* data() const { return data_; }

    TensorNode* grad() { return grad_; }
    const TensorNode* grad() const { return grad_; }
    bool has_grad() const { return grad_ != nullptr; }

    // Gradient requirement
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires) { requires_grad_ = requires; }

    // Autograd: NNGraph-level producer (not TensorGraph)
    bool is_leaf() const { return producer_ == nullptr; }
    bool has_producer() const { return producer_ != nullptr; }
    OpNode* producer() { return producer_; }
    const OpNode* producer() const { return producer_; }

    // Set by NNGraph op that created this tensor (may use multiple TensorGraph ops)
    void set_producer(OpNode* op);

    //! Autograd: propagate upstream gradient through the computation graph.
    //! Grad must be set beforehand (get_or_create_grad + fill/bind).
    //! get_or_create_grad does NOT add CLEAR; backward ops use beta for
    //! overwrite vs accumulate based on the is_first flag.
    //! Does NOT fill grad with ones - user must provide upstream gradient.
    //! When retain_graph is false (default), clears op_nodes_ and producer_
    //! after backward so backward cannot be called again.
    void backward(bool retain_graph = false);

    // Graph access (for operations that deduce graph from tensor)
    NNGraph* graph();

    // Input/output marking (forwarded to data tensor for TensorGraph ops)
    bool is_input() const { return data_ ? data_->is_input() : false; }
    void mark_input(bool v = true)
    {
        if(data_) data_->mark_input(v);
    }
    bool is_output() const { return data_ ? data_->is_output() : false; }
    void mark_output(bool v = true)
    {
        if(data_) data_->mark_output(v);
    }

    // String representation
    std::string to_string() const;

private:
    void set_grad(TensorNode* grad) { grad_ = grad; }
};

} // namespace nntile::graph
