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
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

// Include third-party headers

// Include other NNTile headers
#include <nntile/base_types.hh>

namespace nntile::graph
{

//! Data types supported
enum class DataType
{
    FP32,
    FP32_FAST_TF32,
    FP32_FAST_FP16,
    FP32_FAST_BF16,
    FP64,
    FP16,
    BF16,
    INT64,
    INT32,
    BOOL
};

//! Convert DataType to string
std::string dtype_to_string(DataType dtype);

//! Get size in bytes for DataType
size_t dtype_size(DataType dtype);

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

//! Logical graph - defines computation without physical details
class LogicalGraph
{
public:
    //! Unique identifier for nodes
    using NodeId = uint64_t;

    //! An operation node in the logical graph (forward declaration)
    class OpNode;

    //! A tensor node in the logical graph (forward declaration)
    class TensorNode;

    //! A tensor node in the logical graph (implementation)
    class TensorNode
    {
        friend class LogicalGraph;
        friend class OpNode;

    private:
        NodeId id_;
        LogicalGraph* graph_;
        std::vector<Index> shape_;
        DataType dtype_;
        std::string name_;
        bool is_input_ = false;
        bool is_output_ = false;

        // Graph edges
        // Op that creates this tensor (nullptr if input)
        OpNode* producer_ = nullptr;
        // Ops that use this tensor
        std::vector<OpNode*> consumers_;

    public:
        TensorNode(
            NodeId id,
            LogicalGraph* graph,
            std::vector<Index> shape,
            DataType dtype,
            const std::string& name = ""
        );

        // Accessors
        NodeId id() const { return id_; }
        const std::string& name() const { return name_; }
        DataType dtype() const { return dtype_; }
        const std::vector<Index>& shape() const { return shape_; }
        Index ndim() const { return static_cast<Index>(shape_.size()); }
        Index dim(int idx) const;
        Index nelems() const;
        size_t size_bytes() const;
        bool is_compatible(const TensorNode& other) const;

        // Graph access
        LogicalGraph& graph() { return *graph_; }
        const LogicalGraph& graph() const { return *graph_; }

        // Graph structure
        bool has_producer() const { return producer_ != nullptr; }
        OpNode* producer() const { return producer_; }
        const std::vector<OpNode*>& consumers() const { return consumers_; }
        bool is_input() const { return is_input_; }
        bool is_output() const { return is_output_; }
        void mark_input(bool is_input = true) { is_input_ = is_input; }
        void mark_output(bool is_output = true) { is_output_ = is_output; }

        // String representation
        std::string to_string() const;

    private:
        // Only LogicalGraph/OpNode can modify edges
        void set_producer(OpNode* op) { producer_ = op; }
        void add_consumer(OpNode* op) { consumers_.push_back(op); }
    };

    //! An operation node in the logical graph (implementation)
    class OpNode
    {
        friend class LogicalGraph;

    private:
        NodeId id_;
        LogicalGraph* graph_;
        OpType type_;
        OpAttrs attrs_;
        std::string name_;

        // Graph edges
        std::vector<TensorNode*> inputs_;
        std::vector<TensorNode*> outputs_;

    public:
        OpNode(
            NodeId id,
            LogicalGraph* graph,
            OpType type,
            OpAttrs attrs,
            const std::vector<TensorNode*>& inputs,
            const std::vector<TensorNode*>& outputs,
            const std::string& name = ""
        );

        // Accessors
        NodeId id() const { return id_; }
        const std::string& name() const { return name_; }
        OpType type() const { return type_; }
        const OpAttrs& attrs() const { return attrs_; }

        // Graph access
        LogicalGraph& graph() { return *graph_; }
        const LogicalGraph& graph() const { return *graph_; }

        // Graph structure
        const std::vector<TensorNode*>& inputs() const
        {
            return inputs_;
        }
        const std::vector<TensorNode*>& outputs() const
        {
            return outputs_;
        }

        // Convenience accessors for common cases
        TensorNode* input(size_t idx = 0) const
        {
            return inputs_.at(idx);
        }
        TensorNode* output(size_t idx = 0) const
        {
            return outputs_.at(idx);
        }

        // String representation
        std::string to_string() const;

    private:
        // Only LogicalGraph can modify
        void add_input(TensorNode* t);
        void add_output(TensorNode* t);
    };

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
    TensorNode& tensor(
        std::vector<Index> shape,
        const std::string& name,
        DataType dtype = DataType::FP32);

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
        const std::vector<TensorNode*>& outputs,
        const std::string& name = ""
    );

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
