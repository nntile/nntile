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
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

// Include third-party headers

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor_spec.hh>

namespace nntile::graph
{

// Forward declarations
class LogicalGraph;
class OpNode;

//! Unique identifier for nodes
using NodeId = uint64_t;

//! Operation types
enum class OpType {
    GEMM,
    GELU,
    GELU_BACKWARD,
    ADD_FIBER,       // Broadcast-add a fiber (1D) tensor along last dimension
    SUM_FIBER        // Sum along all but last dimension (reverse of ADD_FIBER)
    // Add more as needed
};

//! Convert OpType to string
std::string op_type_to_string(OpType type);

//! Operation attributes (parameters that aren't tensors)
struct GemmAttrs
{
    bool trans_a = false;
    bool trans_b = false;
    // For GEMM: C = alpha * A @ B + beta * C
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    Index ndim = 1;  // Number of dimensions used in gemm contraction
    Index batch_ndim = 0;  // Number of last dimensions used for batching
};

struct GeluAttrs
{
    // No attributes for basic gelu
};

struct GeluBackwardAttrs
{
    // No attributes for basic gelu_backward
};

struct AddFiberAttrs
{
    // Broadcast-add bias along the last axis
    // output = input + bias (bias has shape [last_dim])
    Scalar alpha = 1.0;  // Scaling factor for bias
};

struct SumFiberAttrs
{
    // Sum along all dimensions except the last one
    // output[i] = sum(input[..., i]) for all batch dimensions
    Scalar alpha = 1.0;  // Scaling factor
    Scalar beta = 0.0;   // For accumulation: output = alpha * sum + beta * output
};

using OpAttrs = std::variant<GemmAttrs, GeluAttrs, GeluBackwardAttrs,
                             AddFiberAttrs, SumFiberAttrs>;

//! A tensor node in the logical graph
class LogicalGraphTensorNode
{
    friend class LogicalGraph;
    friend class OpNode;

private:
    NodeId id_;
    std::string name_;
    TensorSpec spec_;
    LogicalGraph* graph_;

    // Graph edges
    // Op that creates this tensor (nullptr if input)
    OpNode* producer_ = nullptr;
    std::vector<OpNode*> consumers_;       // Ops that use this tensor

public:
    LogicalGraphTensorNode(
        NodeId id,
        const std::string& name,
        TensorSpec spec,
        LogicalGraph* graph);

    // Accessors
    NodeId id() const { return id_; }
    const std::string& name() const { return name_; }
    const TensorSpec& spec() const { return spec_; }
    DataType dtype() const { return spec_.dtype(); }
    const std::vector<Index>& shape() const { return spec_.shape(); }
    Index ndim() const { return spec_.ndim(); }

    // Graph access
    LogicalGraph& graph() { return *graph_; }
    const LogicalGraph& graph() const { return *graph_; }

    // Graph structure
    bool has_producer() const { return producer_ != nullptr; }
    OpNode* producer() const { return producer_; }
    const std::vector<OpNode*>& consumers() const { return consumers_; }

    // String representation
    std::string to_string() const;

private:
    // Only LogicalGraph/OpNode can modify edges
    void set_producer(OpNode* op) { producer_ = op; }
    void add_consumer(OpNode* op) { consumers_.push_back(op); }
    void remove_consumer(OpNode* op);
    void clear_producer() { producer_ = nullptr; }
};

//! Backward-compatible alias
using TensorNode = LogicalGraphTensorNode;

//! An operation node in the logical graph
class OpNode
{
    friend class LogicalGraph;

private:
    NodeId id_;
    OpType type_;
    OpAttrs attrs_;
    LogicalGraph* graph_;

    // Graph edges
    std::vector<LogicalGraphTensorNode*> inputs_;
    std::vector<LogicalGraphTensorNode*> outputs_;

public:
    OpNode(NodeId id, OpType type, OpAttrs attrs, LogicalGraph* graph);

    // Accessors
    NodeId id() const { return id_; }
    OpType type() const { return type_; }
    const OpAttrs& attrs() const { return attrs_; }

    // Graph access
    LogicalGraph& graph() { return *graph_; }
    const LogicalGraph& graph() const { return *graph_; }

    // Graph structure
    const std::vector<LogicalGraphTensorNode*>& inputs() const
    {
        return inputs_;
    }
    const std::vector<LogicalGraphTensorNode*>& outputs() const
    {
        return outputs_;
    }

    // Convenience accessors for common cases
    LogicalGraphTensorNode* input(size_t idx = 0) const
    {
        return inputs_.at(idx);
    }
    LogicalGraphTensorNode* output(size_t idx = 0) const
    {
        return outputs_.at(idx);
    }

    // String representation
    std::string to_string() const;

private:
    // Only LogicalGraph can modify
    void add_input(LogicalGraphTensorNode* t);
    void add_output(LogicalGraphTensorNode* t);
};

//! Logical graph - defines computation without physical details
class LogicalGraph
{
private:
    std::string name_;
    std::vector<std::unique_ptr<LogicalGraphTensorNode>> tensors_;
    std::vector<std::unique_ptr<OpNode>> ops_;
    std::map<std::string, LogicalGraphTensorNode*> tensor_by_name_;

    NodeId next_tensor_id_ = 0;
    NodeId next_op_id_ = 0;

public:
    explicit LogicalGraph(const std::string& name = "");

    // -----------------------------------------------------------------
    // Tensor Creation
    // -----------------------------------------------------------------

    //! Create an input tensor (not produced by any operation)
    LogicalGraphTensorNode& tensor(const TensorSpec& spec, const std::string& name);

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
        const std::vector<LogicalGraphTensorNode*>& inputs,
        const std::vector<LogicalGraphTensorNode*>& outputs
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
    bool can_remove_tensor(const LogicalGraphTensorNode* tensor) const;

    //! Remove a tensor from the graph
    //! The tensor can only be removed if it is not used in any operation
    //! (no producer and no consumers).
    //! @param tensor Pointer to the tensor node to remove
    //! @throws std::invalid_argument if tensor doesn't belong to this graph
    //! @throws std::runtime_error if tensor is used by any operation
    void remove_tensor(LogicalGraphTensorNode* tensor);

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
    LogicalGraphTensorNode* get_tensor(const std::string& name);
    const LogicalGraphTensorNode* get_tensor(const std::string& name) const;

    //! Get all tensor names
    std::vector<std::string> tensor_names() const;

    //! Get all tensors (for iteration)
    const std::vector<std::unique_ptr<LogicalGraphTensorNode>>& tensors() const
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
