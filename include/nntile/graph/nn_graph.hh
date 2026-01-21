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

//! A tensor node in NNGraph, holding data and gradient logical nodes.
class NNGraphTensorNode
{
    friend class NNGraph;

private:
    LogicalGraphTensorNode* data_ = nullptr;
    LogicalGraphTensorNode* grad_ = nullptr;
    bool requires_grad_ = false;

public:
    NNGraphTensorNode(LogicalGraphTensorNode* data, bool requires_grad);

    // Accessors for underlying logical nodes
    LogicalGraphTensorNode& data() { return *data_; }
    const LogicalGraphTensorNode& data() const { return *data_; }
    LogicalGraphTensorNode* data_ptr() { return data_; }
    const LogicalGraphTensorNode* data_ptr() const { return data_; }

    LogicalGraphTensorNode* grad() { return grad_; }
    const LogicalGraphTensorNode* grad() const { return grad_; }
    bool has_grad() const { return grad_ != nullptr; }

    // Gradient requirement
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires) { requires_grad_ = requires; }

    // Convenience accessors (forwarded to data tensor)
    const std::string& name() const { return data_->name(); }
    const TensorSpec& spec() const { return data_->spec(); }
    DataType dtype() const { return data_->dtype(); }
    const std::vector<Index>& shape() const { return data_->shape(); }
    Index ndim() const { return data_->ndim(); }

    // String representation
    std::string to_string() const;

private:
    void set_grad(LogicalGraphTensorNode* grad) { grad_ = grad; }
};

//! Neural network graph - wraps logical graph and tracks gradients
class NNGraph
{
private:
    std::string name_;
    LogicalGraph logical_;
    std::vector<std::unique_ptr<NNGraphTensorNode>> tensors_;
    std::map<std::string, NNGraphTensorNode*> tensor_by_name_;

public:
    explicit NNGraph(const std::string& name = "");

    // -----------------------------------------------------------------
    // Tensor Creation
    // -----------------------------------------------------------------

    //! Create an input tensor (not produced by any operation)
    NNGraphTensorNode& tensor(
        const TensorSpec& spec,
        const std::string& name,
        bool requires_grad = false);

    // -----------------------------------------------------------------
    // Operation Builder API (forward or gradient ops)
    // -----------------------------------------------------------------

    //! Add an operation to the graph using NNGraph tensor nodes
    void add_op(
        OpType type,
        OpAttrs attrs,
        const std::vector<NNGraphTensorNode*>& inputs,
        const std::vector<NNGraphTensorNode*>& outputs
    );

    //! Add an operation using underlying logical tensor nodes
    void add_op(
        OpType type,
        OpAttrs attrs,
        const std::vector<LogicalGraphTensorNode*>& inputs,
        const std::vector<LogicalGraphTensorNode*>& outputs
    );

    // -----------------------------------------------------------------
    // Removal API
    // -----------------------------------------------------------------

    bool can_remove_op(const OpNode* op) const { return logical_.can_remove_op(op); }
    void remove_op(OpNode* op) { logical_.remove_op(op); }

    bool can_remove_tensor(const NNGraphTensorNode* tensor) const;
    void remove_tensor(NNGraphTensorNode* tensor);
    void remove_tensor(const std::string& name);

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return logical_.num_ops(); }

    NNGraphTensorNode* get_tensor(const std::string& name);
    const NNGraphTensorNode* get_tensor(const std::string& name) const;

    std::vector<std::string> tensor_names() const;

    const std::vector<std::unique_ptr<NNGraphTensorNode>>& tensors() const
    {
        return tensors_;
    }
    const std::vector<std::unique_ptr<OpNode>>& ops() const
    {
        return logical_.ops();
    }

    LogicalGraph& logical_graph() { return logical_; }
    const LogicalGraph& logical_graph() const { return logical_; }

    // -----------------------------------------------------------------
    // Gradient helpers
    // -----------------------------------------------------------------

    bool requires_grad(const NNGraphTensorNode& tensor) const;
    void set_requires_grad(NNGraphTensorNode& tensor, bool requires = true);

    bool is_first_grad(const NNGraphTensorNode& tensor) const
    {
        return tensor.grad() == nullptr;
    }

    LogicalGraphTensorNode& get_or_create_grad(
        NNGraphTensorNode& tensor,
        const std::string& grad_name);

    // -----------------------------------------------------------------
    // String representation
    // -----------------------------------------------------------------

    std::string to_string() const;
};

} // namespace nntile::graph
