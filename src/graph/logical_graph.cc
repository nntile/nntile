/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical_graph.cc
 * Implementation of LogicalGraph class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical_graph.hh"

// Include standard headers
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

// Include third-party headers

// Include other NNTile headers

namespace nntile::graph
{

//! Convert DataType to string
std::string dtype_to_string(DataType dtype)
{
    switch(dtype)
    {
        case DataType::FP32:
            return "FP32";
        case DataType::FP32_FAST_TF32:
            return "FP32_FAST_TF32";
        case DataType::FP32_FAST_FP16:
            return "FP32_FAST_FP16";
        case DataType::FP32_FAST_BF16:
            return "FP32_FAST_BF16";
        case DataType::FP64:
            return "FP64";
        case DataType::FP16:
            return "FP16";
        case DataType::BF16:
            return "BF16";
        case DataType::INT64:
            return "INT64";
        case DataType::INT32:
            return "INT32";
        case DataType::BOOL:
            return "BOOL";
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

//! Get size in bytes for DataType
size_t dtype_size(DataType dtype)
{
    switch(dtype)
    {
        // 1 byte
        case DataType::BOOL:
            return 1;
        // 2 bytes
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        // 4 bytes
        case DataType::FP32:
        case DataType::FP32_FAST_TF32:
        case DataType::FP32_FAST_FP16:
        case DataType::FP32_FAST_BF16:
        case DataType::INT32:
            return 4;
        // 8 bytes
        case DataType::FP64:
        case DataType::INT64:
            return 8;
        default:
            throw std::invalid_argument("Unknown DataType");
    }
}

//! Convert OpType to string
std::string op_type_to_string(OpType type)
{
    switch(type)
    {
        case OpType::GEMM:
            return "GEMM";
        case OpType::GELU:
            return "GELU";
        case OpType::GELU_BACKWARD:
            return "GELU_BACKWARD";
        case OpType::ADD_FIBER:
            return "ADD_FIBER";
        case OpType::SUM_FIBER:
            return "SUM_FIBER";
        case OpType::CLEAR:
            return "CLEAR";

        // Element-wise unary operations
        case OpType::GELU_INPLACE:
            return "GELU_INPLACE";
        case OpType::GELUTANH:
            return "GELUTANH";
        case OpType::GELUTANH_INPLACE:
            return "GELUTANH_INPLACE";
        case OpType::GELUTANH_BACKWARD:
            return "GELUTANH_BACKWARD";
        case OpType::RELU:
            return "RELU";
        case OpType::RELU_INPLACE:
            return "RELU_INPLACE";
        case OpType::RELU_BACKWARD:
            return "RELU_BACKWARD";
        case OpType::SILU:
            return "SILU";
        case OpType::SILU_INPLACE:
            return "SILU_INPLACE";
        case OpType::SILU_BACKWARD:
            return "SILU_BACKWARD";
        case OpType::SQRT:
            return "SQRT";
        case OpType::SQRT_INPLACE:
            return "SQRT_INPLACE";
        case OpType::HYPOT:
            return "HYPOT";
        case OpType::HYPOT_INPLACE:
            return "HYPOT_INPLACE";

        // Element-wise binary operations
        case OpType::ADD:
            return "ADD";
        case OpType::ADD_INPLACE:
            return "ADD_INPLACE";
        case OpType::MULTIPLY:
            return "MULTIPLY";
        case OpType::MULTIPLY_INPLACE:
            return "MULTIPLY_INPLACE";
        case OpType::HYPOT_SCALAR_INVERSE:
            return "HYPOT_SCALAR_INVERSE";
        case OpType::SUBTRACT_INDEXED_OUTPUTS:
            return "SUBTRACT_INDEXED_OUTPUTS";

        // Reduction operations
        case OpType::SUM:
            return "SUM";
        case OpType::SUM_SLICE:
            return "SUM_SLICE";
        case OpType::NORM:
            return "NORM";
        case OpType::NORM_FIBER:
            return "NORM_FIBER";
        case OpType::NORM_FIBER_INPLACE:
            return "NORM_FIBER_INPLACE";
        case OpType::NORM_SLICE:
            return "NORM_SLICE";
        case OpType::NORM_SLICE_INPLACE:
            return "NORM_SLICE_INPLACE";
        case OpType::LOGSUMEXP:
            return "LOGSUMEXP";
        case OpType::MAXSUMEXP:
            return "MAXSUMEXP";
        case OpType::SUMPROD_FIBER:
            return "SUMPROD_FIBER";
        case OpType::SUMPROD_SLICE:
            return "SUMPROD_SLICE";

        // Scale operations
        case OpType::SCALE:
            return "SCALE";
        case OpType::SCALE_INPLACE:
            return "SCALE_INPLACE";
        case OpType::SCALE_FIBER:
            return "SCALE_FIBER";
        case OpType::SCALE_SLICE:
            return "SCALE_SLICE";

        // Add operations
        case OpType::ADD_FIBER_INPLACE:
            return "ADD_FIBER_INPLACE";
        case OpType::ADD_SLICE:
            return "ADD_SLICE";
        case OpType::ADD_SLICE_INPLACE:
            return "ADD_SLICE_INPLACE";

        // Matrix operations
        case OpType::TRANSPOSE:
            return "TRANSPOSE";

        // Convolution operations
        case OpType::CONV2D_INPLACE:
            return "CONV2D_INPLACE";
        case OpType::CONV2D_BWD_INPUT_INPLACE:
            return "CONV2D_BWD_INPUT_INPLACE";
        case OpType::CONV2D_BWD_WEIGHT_INPLACE:
            return "CONV2D_BWD_WEIGHT_INPLACE";

        // Embedding operations
        case OpType::EMBEDDING:
            return "EMBEDDING";
        case OpType::EMBEDDING_BACKWARD:
            return "EMBEDDING_BACKWARD";

        // Mixed-dtype operations
        case OpType::MASK_SCALAR:
            return "MASK_SCALAR";
        case OpType::TOTAL_SUM_ACCUM:
            return "TOTAL_SUM_ACCUM";

        // Optimizer operations
        case OpType::SGD_STEP:
            return "SGD_STEP";
        case OpType::ADAM_STEP:
            return "ADAM_STEP";
        case OpType::ADAMW_STEP:
            return "ADAMW_STEP";

        // Utility operations
        case OpType::COPY:
            return "COPY";
        case OpType::COPY_INTERSECTION:
            return "COPY_INTERSECTION";
        case OpType::GATHER:
            return "GATHER";
        case OpType::SCATTER:
            return "SCATTER";
        case OpType::FILL:
            return "FILL";
        case OpType::POW:
            return "POW";
        case OpType::POW_INPLACE:
            return "POW_INPLACE";
        case OpType::LOG_SCALAR:
            return "LOG_SCALAR";

        // Random operations
        case OpType::RANDN:
            return "RANDN";

        // Flash attention (CUDA-only)
        case OpType::FLASH_SDPA_FWD_CUDNN:
            return "FLASH_SDPA_FWD_CUDNN";
        case OpType::FLASH_SDPA_BWD_CUDNN:
            return "FLASH_SDPA_BWD_CUDNN";

        // Rotary position embedding
        case OpType::ROPE:
            return "ROPE";
        case OpType::ROPE_BACKWARD:
            return "ROPE_BACKWARD";

        default:
            throw std::invalid_argument("Unknown OpType");
    }
}

//! A tensor node in the logical graph
LogicalGraph::TensorNode::TensorNode(
    NodeId id,
    LogicalGraph* graph,
    std::vector<Index> shape,
    DataType dtype,
    const std::string& name
)
    : id_(id)
    , graph_(graph)
    , shape_(std::move(shape))
    , dtype_(dtype)
    , name_(name)
{
    for(Index dim : shape_)
    {
        if(dim <= 0)
        {
            throw std::invalid_argument(
                "TensorNode: all dimensions must be positive");
        }
    }
}

//! Get dimension at index (supports negative indexing)
Index LogicalGraph::TensorNode::dim(int idx) const
{
    if(idx < 0)
    {
        idx += static_cast<int>(shape_.size());
    }
    if(idx < 0 || static_cast<size_t>(idx) >= shape_.size())
    {
        throw std::out_of_range("TensorNode::dim: index out of range");
    }
    return shape_[static_cast<size_t>(idx)];
}

//! Total number of elements
Index LogicalGraph::TensorNode::nelems() const
{
    return std::accumulate(shape_.begin(), shape_.end(), Index(1),
        std::multiplies<Index>());
}

//! Total size in bytes
size_t LogicalGraph::TensorNode::size_bytes() const
{
    return static_cast<size_t>(nelems()) * dtype_size(dtype_);
}

//! Check if tensor specs are compatible for operations
bool LogicalGraph::TensorNode::is_compatible(const TensorNode& other) const
{
    return dtype_ == other.dtype_;
}

//! String representation
std::string LogicalGraph::TensorNode::to_string() const
{
    std::string result = "LogicalGraph::TensorNode(id=" +
        std::to_string(id_) + ", name='" + name_ + "', shape=[";
    for(size_t i = 0; i < shape_.size(); ++i)
    {
        if(i > 0)
        {
            result += ", ";
        }
        result += std::to_string(shape_[i]);
    }
    result += "], dtype=" + dtype_to_string(dtype_) + ")";
    return result;
}

//! Remove a consumer from this tensor's consumer list
//! An operation node in the logical graph
LogicalGraph::OpNode::OpNode(
    NodeId id,
    LogicalGraph* graph,
    OpType type,
    OpAttrs attrs,
    const std::vector<TensorNode*>& inputs,
    const std::vector<TensorNode*>& outputs,
    const std::string& name
)
    : id_(id)
    , graph_(graph)
    , type_(type)
    , attrs_(std::move(attrs))
    , name_(name)
{
    for(auto* input : inputs)
    {
        add_input(input);
    }
    for(auto* output : outputs)
    {
        add_output(output);
    }
}

//! String representation
std::string LogicalGraph::OpNode::to_string() const
{
    std::string result = op_type_to_string(type_) + "(id=" +
        std::to_string(id_);
    if(!name_.empty())
    {
        result += ", name='" + name_ + "'";
    }
    result += ", inputs=[";
    for(size_t i = 0; i < inputs_.size(); ++i)
    {
        if(i > 0)
        {
            result += ", ";
        }
        result += inputs_[i]->name();
    }
    result += "], outputs=[";
    for(size_t i = 0; i < outputs_.size(); ++i)
    {
        if(i > 0)
        {
            result += ", ";
        }
        result += outputs_[i]->name();
    }
    result += "])";
    return result;
}

//! Only LogicalGraph can modify
void LogicalGraph::OpNode::add_input(TensorNode* t)
{
    inputs_.push_back(t);
    t->add_consumer(this);
}

//! Only LogicalGraph can modify
void LogicalGraph::OpNode::add_output(TensorNode* t)
{
    outputs_.push_back(t);
    t->set_producer(this);
}

LogicalGraph::LogicalGraph(const std::string& name)
    : name_(name)
{
}

//! Create an input tensor (not produced by any operation)
LogicalGraph::TensorNode& LogicalGraph::tensor(
    std::vector<Index> shape,
    const std::string& name,
    DataType dtype
)
{
    // Check name doesn't already exist
    if(tensor_by_name_.count(name) > 0)
    {
        throw std::invalid_argument("LogicalGraph::tensor: tensor '" + name +
                "' already exists");
    }

    // Create TensorNode with unique ID
    auto node = std::make_unique<TensorNode>(
        next_tensor_id_,
        this,
        std::move(shape),
        dtype,
        name
    );
    ++next_tensor_id_;
    TensorNode* node_ptr = node.get();

    // Store in containers
    tensors_.push_back(std::move(node));
    tensor_by_name_[name] = node_ptr;

    return *node_ptr;
}

//! Add an operation to the graph with specified output tensors
void LogicalGraph::add_op(
    OpType type,
    OpAttrs attrs,
    const std::vector<TensorNode*>& inputs,
    const std::vector<TensorNode*>& outputs,
    const std::string& name)
{
    // Validate all inputs belong to this graph
    for(const auto* input : inputs)
    {
        if(&input->graph() != this)
        {
            throw std::invalid_argument(
                "LogicalGraph::add_op: input tensor '" + input->name() +
                "' does not belong to this graph");
        }
    }

    // Validate all outputs belong to this graph
    for(const auto* output : outputs)
    {
        if(&output->graph() != this)
        {
            throw std::invalid_argument(
                "LogicalGraph::add_op: output tensor '" + output->name() +
                "' does not belong to this graph");
        }
    }

    // Create OpNode
    auto op = std::make_unique<OpNode>(
        next_op_id_,
        this,
        type,
        attrs,
        inputs,
        outputs,
        name
    );
    ++next_op_id_;
    OpNode* op_ptr = op.get();

    // Store operation
    ops_.push_back(std::move(op));
}

//! Get tensor by name (returns nullptr if not found)
LogicalGraph::TensorNode* LogicalGraph::get_tensor(const std::string& name)
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get tensor by name (returns nullptr if not found)
const LogicalGraph::TensorNode* LogicalGraph::get_tensor(
    const std::string& name) const
{
    auto it = tensor_by_name_.find(name);
    return it != tensor_by_name_.end() ? it->second : nullptr;
}

//! Get all tensor names
std::vector<std::string> LogicalGraph::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensor_by_name_.size());
    for(const auto& pair : tensor_by_name_)
    {
        names.push_back(pair.first);
    }
    return names;
}

//! String representation
std::string LogicalGraph::to_string() const
{
    std::stringstream ss;
    ss << "LogicalGraph(name='" << name_ << "', tensors=" << num_tensors()
       << ", ops=" << num_ops() << ")\n";

    ss << "Tensors:\n";
    for(const auto& t : tensors_)
    {
        ss << "  " << t->to_string() << "\n";
    }

    ss << "Operations:\n";
    for(const auto& op : ops_)
    {
        ss << "  " << op->to_string() << "\n";
    }

    return ss.str();
}

} // namespace nntile::graph
