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
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
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
    // Existing operations
    GEMM,
    GELU,
    GELU_BACKWARD,
    ADD_FIBER,
    SUM_FIBER,
    CLEAR,

    // Element-wise unary operations
    GELU_INPLACE,
    GELUTANH,
    GELUTANH_INPLACE,
    GELUTANH_BACKWARD,
    RELU,
    RELU_INPLACE,
    RELU_BACKWARD,
    SILU,
    SILU_INPLACE,
    SILU_BACKWARD,
    SOFTMAX,
    SOFTMAX_INPLACE,
    SQRT,
    SQRT_INPLACE,
    HYPOT,
    HYPOT_INPLACE,

    // Element-wise binary operations
    ADD,
    ADD_INPLACE,
    MULTIPLY,
    MULTIPLY_INPLACE,
    HYPOT_SCALAR_INVERSE,
    SUBTRACT_INDEXED_OUTPUTS,

    // Reduction operations
    SUM,
    SUM_SLICE,
    NORM,
    NORM_FIBER,
    NORM_FIBER_INPLACE,
    NORM_SLICE,
    NORM_SLICE_INPLACE,
    LOGSUMEXP,
    MAXSUMEXP,
    SUMPROD_FIBER,
    SUMPROD_SLICE,

    // Scale operations
    SCALE,
    SCALE_INPLACE,
    SCALE_FIBER,
    SCALE_SLICE,

    // Add operations
    ADD_FIBER_INPLACE,
    ADD_SLICE,
    ADD_SLICE_INPLACE,

    // Multiply operations
    MULTIPLY_FIBER,
    MULTIPLY_FIBER_INPLACE,
    MULTIPLY_SLICE,

    // Matrix operations
    TRANSPOSE,

    // Convolution operations
    CONV2D_INPLACE,
    CONV2D_BWD_INPUT_INPLACE,
    CONV2D_BWD_WEIGHT_INPLACE,

    // Embedding operations
    EMBEDDING,
    EMBEDDING_BACKWARD,

    // Mixed-dtype operations
    MASK_SCALAR,
    TOTAL_SUM_ACCUM,

    // Optimizer operations
    SGD_STEP,
    ADAM_STEP,
    ADAMW_STEP,

    // Utility operations
    COPY,
    COPY_INTERSECTION,
    GATHER,
    SCATTER,
    FILL,
    POW,
    POW_INPLACE,
    LOG_SCALAR,

    // Random operations
    RANDN,

    // Flash attention (CUDA-only)
    FLASH_SDPA_FWD_CUDNN,
    FLASH_SDPA_BWD_CUDNN,

    // Rotary position embedding
    ROPE,
    ROPE_BACKWARD
};

//! Convert OpType to string
std::string op_type_to_string(OpType type);

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
        std::shared_ptr<void> attrs_;
        std::string name_;

        // Graph edges
        std::vector<TensorNode*> inputs_;
        std::vector<TensorNode*> outputs_;

    public:
        OpNode(
            NodeId id,
            LogicalGraph* graph,
            OpType type,
            std::shared_ptr<void> attrs,
            const std::vector<TensorNode*>& inputs,
            const std::vector<TensorNode*>& outputs,
            const std::string& name = ""
        );

        // Accessors
        NodeId id() const { return id_; }
        const std::string& name() const { return name_; }
        OpType type() const { return type_; }
        //! Opaque attrs; cast in op impl: std::static_pointer_cast<MyAttrs>(op->attrs())
        const std::shared_ptr<void>& attrs() const { return attrs_; }

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
    //! @param attrs Opaque (std::shared_ptr<void>); only op impl knows the type
    //! @param inputs Vector of input tensor pointers (must belong to this graph)
    //! @param outputs Vector of output tensor pointers (must belong to this graph)
    void add_op(
        OpType type,
        std::shared_ptr<void> attrs,
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

    //! Generate mermaid graph visualization
    std::string to_mermaid() const;
};

} // namespace nntile::graph
