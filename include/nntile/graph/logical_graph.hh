/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical_graph.hh
 * LogicalGraph class for defining computation graphs.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <map>
#include <memory>
#include <string>
#include <vector>

// Include third-party headers

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/op_node.hh>
#include <nntile/graph/tensor_node.hh>

namespace nntile::graph
{

//! Logical graph - defines computation without physical details
class LogicalGraph
{
private:
    std::string name_;
    std::vector<std::unique_ptr<TensorNode>> tensors_;
    std::vector<std::unique_ptr<OpNode>> ops_;
    std::map<std::string, TensorNode*> tensor_by_name_;

    NodeId next_tensor_id_ = 0;
    NodeId next_op_id_ = 0;

public:
    explicit LogicalGraph(const std::string& name = "");

    // -----------------------------------------------------------------
    // Tensor Creation
    // -----------------------------------------------------------------

    //! Create an input tensor (not produced by any operation)
    TensorNode& tensor(const TensorSpec& spec, const std::string& name);

    // -----------------------------------------------------------------
    // Operation Builder API (used by free functions to add operations)
    // -----------------------------------------------------------------

    //! Add an operation to the graph with specified output tensors
    //! This is the public builder API for operation implementations.
    //! @param type The operation type
    //! @param attrs The operation attributes
    //! @param inputs Vector of input tensor pointers (must belong to this graph)
    //! @param outputs Vector of output tensor pointers (must belong to this graph)
    void add_op(
        OpType type,
        OpAttrs attrs,
        const std::vector<TensorNode*>& inputs,
        const std::vector<TensorNode*>& outputs
    );

    // -----------------------------------------------------------------
    // Removal API
    // -----------------------------------------------------------------

    //! Check if an operation can be removed (no other ops depend on its outputs)
    //! @param op Pointer to the operation node
    //! @return true if the operation can be safely removed
    bool can_remove_op(const OpNode* op) const;

    //! Remove an operation from the graph
    //! The operation can only be removed if no other operation depends on its
    //! outputs (i.e., output tensors have no consumers other than this op).
    //! This disconnects the op from its input/output tensors but does NOT
    //! remove the output tensors themselves.
    //! @param op Pointer to the operation node to remove
    //! @throws std::invalid_argument if op doesn't belong to this graph
    //! @throws std::runtime_error if operation has dependent operations
    void remove_op(OpNode* op);

    //! Check if a tensor can be removed (not used in any operation)
    //! @param tensor Pointer to the tensor node
    //! @return true if the tensor can be safely removed
    bool can_remove_tensor(const TensorNode* tensor) const;

    //! Remove a tensor from the graph
    //! The tensor can only be removed if it is not used in any operation
    //! (no producer and no consumers).
    //! @param tensor Pointer to the tensor node to remove
    //! @throws std::invalid_argument if tensor doesn't belong to this graph
    //! @throws std::runtime_error if tensor is used by any operation
    void remove_tensor(TensorNode* tensor);

    //! Remove a tensor by name
    //! @param name Name of the tensor to remove
    //! @throws std::invalid_argument if tensor not found
    //! @throws std::runtime_error if tensor is used by any operation
    void remove_tensor(const std::string& name);

    // -----------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------

    const std::string& name() const { return name_; }
    size_t num_tensors() const { return tensors_.size(); }
    size_t num_ops() const { return ops_.size(); }

    //! Get tensor by name (returns nullptr if not found)
    TensorNode* get_tensor(const std::string& name);
    const TensorNode* get_tensor(const std::string& name) const;

    //! Get all tensor names
    std::vector<std::string> tensor_names() const;

    //! Get all tensors (for iteration)
    const std::vector<std::unique_ptr<TensorNode>>& tensors() const
    {
        return tensors_;
    }

    //! Get all ops (for iteration)
    const std::vector<std::unique_ptr<OpNode>>& ops() const
    {
        return ops_;
    }

    //! String representation
    std::string to_string() const;
};

} // namespace nntile::graph
