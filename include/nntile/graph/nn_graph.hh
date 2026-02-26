/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn_graph.hh
 * NNGraph class for defining computation graphs with gradients.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <map>
#include <memory>
#include <string>
#include <vector>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Neural network graph - wraps logical graph and tracks gradients
class NNGraph
{
public:
    //! A tensor node in NNGraph, holding data and gradient tensor node.
    //! Mimics PyTorch tensor with autograd: grad_fn points to the op that
    //! produced this tensor; backward() builds the gradient graph.
    class TensorNode
    {
        friend class NNGraph;

    private:
        NNGraph* graph_ = nullptr;
        LogicalGraph::TensorNode* data_ = nullptr;
        // Gradient is itself an NNGraph::TensorNode for convenience
        TensorNode* grad_ = nullptr;
        bool requires_grad_ = true;

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

        // Autograd: producer op (grad_fn) - nullptr for leaf/input tensors
        LogicalGraph::OpNode* grad_fn() const
        {
            return data_ ? data_->producer() : nullptr;
        }
        bool is_leaf() const { return grad_fn() == nullptr; }

        // Autograd: build backward graph from this tensor (PyTorch-style).
        //! Initializes grad with ones if not set, then propagates through
        //! the computation graph.
        void backward();

        // Convenience accessors (forwarded to data tensor)
        const std::string& name() const { return data_->name(); }
        DataType dtype() const { return data_->dtype(); }
        const std::vector<Index>& shape() const { return data_->shape(); }
        Index ndim() const { return data_->ndim(); }

        // Input/output marking (forwarded to data tensor)
        // Mark as graph input (provided via bind_data) - never invalidated
        bool is_input() const { return data_->is_input(); }
        void mark_input(bool is_input = true) { data_->mark_input(is_input); }
        // Mark as graph output (retrieved via get_output) - never invalidated
        bool is_output() const { return data_->is_output(); }
        void mark_output(bool is_output = true) { data_->mark_output(is_output); }

        // String representation
        std::string to_string() const;

    private:
        void set_grad(TensorNode* grad) { grad_ = grad; }
        void set_graph(NNGraph* graph) { graph_ = graph; }
    };

private:
    std::string name_;
    LogicalGraph logical_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::map<std::string, TensorNode*> tensor_by_name_;

public:
    explicit NNGraph(const std::string& name = "");

    // -----------------------------------------------------------------
    // Tensor Creation
    // -----------------------------------------------------------------

    //! Create an input tensor (not produced by any operation)
    TensorNode& tensor(
        std::vector<Index> shape,
        const std::string& name,
        DataType dtype = DataType::FP32,
        bool requires_grad = true
    );

    //! Create NNGraph wrapper for an existing LogicalGraph tensor.
    //! The tensor must belong to this graph's logical graph.
    TensorNode& tensor(LogicalGraph::TensorNode& data,
                      bool requires_grad = false);

    // -----------------------------------------------------------------
    // Operation Builder API (forward or gradient ops)
    // -----------------------------------------------------------------

    //! Add an operation to the graph using NNGraph tensor nodes
    void add_op(
        OpType type,
        OpAttrs attrs,
        const std::vector<TensorNode*>& inputs,
        const std::vector<TensorNode*>& outputs,
        const std::string& name = ""
    );

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return logical_.num_ops(); }

    TensorNode* get_tensor(const std::string& name);
    const TensorNode* get_tensor(const std::string& name) const;

    std::vector<std::string> tensor_names() const;

    const std::vector<std::unique_ptr<TensorNode>>& tensors() const
    {
        return tensors_;
    }
    const std::vector<std::unique_ptr<LogicalGraph::OpNode>>& ops() const
    {
        return logical_.ops();
    }

    LogicalGraph& logical_graph() { return logical_; }
    const LogicalGraph& logical_graph() const { return logical_; }

    // -----------------------------------------------------------------
    // Gradient helpers
    // -----------------------------------------------------------------

    bool requires_grad(const TensorNode& tensor) const;
    void set_requires_grad(TensorNode& tensor, bool requires = true);

    bool is_first_grad(const TensorNode& tensor) const
    {
        return tensor.grad() == nullptr;
    }

    TensorNode& get_or_create_grad(
        TensorNode& tensor,
        const std::string& grad_name);

    // -----------------------------------------------------------------
    // String representation
    // -----------------------------------------------------------------

    std::string to_string() const;

    //! Generate mermaid graph visualization (delegates to logical graph)
    std::string to_mermaid() const { return logical_.to_mermaid(); }
};

} // namespace nntile::graph
