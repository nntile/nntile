/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/linear.cc
 * Linear module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/linear.hh"

// Include standard headers
#include <cstring>
#include <stdexcept>

// Include NNTile headers
#include "nntile/graph/dtype.hh"

namespace nntile::module
{

namespace
{

constexpr Scalar GEMM_ALPHA = 1.0;
constexpr Index GEMM_NDIM_MATRIX = 1;
constexpr Index NO_BATCH_DIM = 0;
constexpr bool NO_TRANSPOSE = false;
constexpr Scalar ADD_FIBER_ALPHA = 1.0;
constexpr Scalar ADD_FIBER_BETA = 1.0;

} // anonymous namespace

//! Constructor: creates new weight tensor, no bias
Linear::Linear(graph::NNGraph* graph,
               const std::string& name,
               Index input_dim, Index output_dim,
               graph::DataType dtype)
    : Module(graph, name)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Create weight tensor during construction
    weight_tensor_ = graph_->tensor(
        {input_dim_, output_dim_},
        tensor_name("weight"),
        dtype_,
        true);
    register_parameter("weight", weight_tensor_);
}

//! Constructor: creates new weight and optionally bias tensors
Linear::Linear(graph::NNGraph* graph,
               const std::string& name,
               Index input_dim, Index output_dim,
               bool with_bias,
               graph::DataType dtype)
    : Module(graph, name)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Create weight tensor
    weight_tensor_ = graph_->tensor(
        {input_dim_, output_dim_},
        tensor_name("weight"),
        dtype_,
        true);
    register_parameter("weight", weight_tensor_);

    // Create bias tensor if requested
    if(with_bias)
    {
        bias_tensor_ = graph_->tensor(
            {output_dim_},
            tensor_name("bias"),
            dtype_,
            true);
        register_parameter("bias", bias_tensor_);
    }
}

//! Constructor: uses existing weight tensor, no bias
Linear::Linear(graph::NNGraph* graph,
               const std::string& name,
               graph::NNGraph::TensorNode* weight_tensor)
    : Module(graph, name)
    , weight_tensor_(weight_tensor)
    , input_dim_(0)
    , output_dim_(0)
    , dtype_(weight_tensor != nullptr ? weight_tensor->dtype() : graph::DataType::FP32)
{
    if(weight_tensor == nullptr)
    {
        throw std::invalid_argument(
            "Linear::Linear: weight_tensor must be non-null");
    }
    // Validate weight tensor has at least 2 dimensions
    if(weight_tensor->ndim() < 2)
    {
        throw std::invalid_argument(
            "Linear::Linear: weight tensor must have at least 2 dimensions, "
            "got " + std::to_string(weight_tensor->ndim()));
    }

    // Extract dimensions from weight tensor shape
    // Weight shape is [input_dim, output_dim]
    const auto& w_shape = weight_tensor->shape();
    input_dim_ = w_shape[0];
    output_dim_ = w_shape[1];

    register_parameter("weight", weight_tensor_);
}

//! Constructor: uses existing weight and bias tensors
Linear::Linear(graph::NNGraph* graph,
               const std::string& name,
               graph::NNGraph::TensorNode* weight_tensor,
               graph::NNGraph::TensorNode* bias_tensor)
    : Module(graph, name)
    , weight_tensor_(weight_tensor)
    , bias_tensor_(bias_tensor)
    , input_dim_(0)
    , output_dim_(0)
    , dtype_(weight_tensor != nullptr ? weight_tensor->dtype() : graph::DataType::FP32)
{
    if(weight_tensor == nullptr || bias_tensor == nullptr)
    {
        throw std::invalid_argument(
            "Linear::Linear: weight_tensor and bias_tensor must be non-null");
    }
    // Validate weight tensor has at least 2 dimensions
    if(weight_tensor->ndim() < 2)
    {
        throw std::invalid_argument(
            "Linear::Linear: weight tensor must have at least 2 dimensions, "
            "got " + std::to_string(weight_tensor->ndim()));
    }

    // Extract dimensions from weight tensor shape
    // Weight shape is [input_dim, output_dim]
    const auto& w_shape = weight_tensor->shape();
    input_dim_ = w_shape[0];
    output_dim_ = w_shape[1];

    // Validate bias tensor shape
    if(bias_tensor->ndim() != 1)
    {
        throw std::invalid_argument(
            "Linear::Linear: bias tensor must be 1-dimensional, "
            "got " + std::to_string(bias_tensor->ndim()));
    }
    if(bias_tensor->shape()[0] != output_dim_)
    {
        throw std::invalid_argument(
            "Linear::Linear: bias tensor dimension mismatch. "
            "Expected " + std::to_string(output_dim_) + ", got " +
            std::to_string(bias_tensor->shape()[0]));
    }

    // Validate data types match
    if(bias_tensor->dtype() != dtype_)
    {
        throw std::invalid_argument(
            "Linear::Linear: bias tensor dtype must match weight tensor dtype");
    }

    register_parameter("weight", weight_tensor_);
    register_parameter("bias", bias_tensor_);
}

graph::NNGraph::TensorNode* Linear::forward(
    graph::NNGraph::TensorNode* input)
{
    if(input == nullptr)
    {
        throw std::invalid_argument(
            "Linear::forward: input tensor must be non-null");
    }
    if(input->ndim() < 1)
    {
        throw std::invalid_argument(
            "Linear::forward_impl: input tensor must have at least one "
            "dimension, got 0-dimensional (scalar) tensor");
    }
    if(input->shape().back() != input_dim_)
    {
        throw std::invalid_argument(
            "Linear::forward_impl: input tensor feature dimension mismatch. "
            "Expected " + std::to_string(input_dim_) + ", got " +
            std::to_string(input->shape().back()));
    }

    input_tensor_ = input;

    const std::string gemm_name = bias_tensor_ != nullptr
        ? tensor_name("gemm_output")
        : tensor_name("output");
    graph::NNGraph::TensorNode* gemm_out = graph::gemm(
        input,
        weight_tensor_,
        gemm_name,
        GEMM_ALPHA,
        NO_TRANSPOSE,
        NO_TRANSPOSE,
        GEMM_NDIM_MATRIX,
        NO_BATCH_DIM);

    if(bias_tensor_ != nullptr)
    {
        const Index feature_axis = gemm_out->ndim() - 1;
        output_tensor_ = graph::add_fiber(
            ADD_FIBER_ALPHA,
            bias_tensor_,
            ADD_FIBER_BETA,
            gemm_out,
            tensor_name("output"),
            feature_axis,
            NO_BATCH_DIM);
    }
    else
    {
        output_tensor_ = gemm_out;
    }

    return output_tensor_;
}

void Linear::bind_weight(std::vector<std::uint8_t> data)
{
    if(weight_tensor_ == nullptr)
    {
        throw std::runtime_error(
            "Linear::bind_weight: weight tensor is null (external weight mode)");
    }
    weight_tensor_->data()->set_bind_hint(std::move(data));
    weight_tensor_->mark_input(true);
}

void Linear::bind_weight(const std::vector<float>& data)
{
    const size_t expected = static_cast<size_t>(input_dim_) * output_dim_;
    if(data.size() != expected)
    {
        throw std::invalid_argument(
            "Linear::bind_weight: size mismatch, expected " +
            std::to_string(expected) + " elements, got " +
            std::to_string(data.size()));
    }
    std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    bind_weight(std::move(bytes));
}

void Linear::bind_bias(std::vector<std::uint8_t> data)
{
    if(bias_tensor_ == nullptr)
    {
        throw std::runtime_error(
            "Linear::bind_bias: bias tensor is null (no bias)");
    }
    bias_tensor_->data()->set_bind_hint(std::move(data));
    bias_tensor_->mark_input(true);
}

void Linear::bind_bias(const std::vector<float>& data)
{
    const size_t expected = static_cast<size_t>(output_dim_);
    if(data.size() != expected)
    {
        throw std::invalid_argument(
            "Linear::bind_bias: size mismatch, expected " +
            std::to_string(expected) + " elements, got " +
            std::to_string(data.size()));
    }
    std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    bind_bias(std::move(bytes));
}

// -----------------------------------------------------------------
// HF Import/Export
// -----------------------------------------------------------------

void Linear::import_hf(const io::SafeTensorsReader& reader,
                       const std::string& hf_prefix)
{
    const std::string w_name = hf_prefix + ".weight";

    if(!reader.has_tensor(w_name))
    {
        throw std::runtime_error(
            "Linear::import_hf: tensor '" + w_name + "' not found");
    }

    const auto& w_info = reader.tensor_info(w_name);
    auto w_data = reader.read_tensor(w_name);

    // HF stores weight as (out_features, in_features) row-major.
    // NNTile stores as (input_dim, output_dim) column-major.
    // Row-major [out, in] has the same byte layout as column-major [in, out]:
    // element (j, i) in row-major is at offset j*in + i,
    // element (i, j) in column-major is at offset i + j*in = j*in + i.
    // So no byte shuffle is needed -- just reinterpret the dimensions.
    weight_tensor_->data()->set_bind_hint(std::move(w_data));
    weight_tensor_->mark_input(true);

    // Bias: 1D, no conversion needed
    if(bias_tensor_ != nullptr)
    {
        const std::string b_name = hf_prefix + ".bias";
        if(reader.has_tensor(b_name))
        {
            auto b_data = reader.read_tensor(b_name);
            bias_tensor_->data()->set_bind_hint(std::move(b_data));
            bias_tensor_->mark_input(true);
        }
    }
}

void Linear::export_hf(io::SafeTensorsWriter& writer,
                       const std::string& hf_prefix) const
{
    const std::string w_name = hf_prefix + ".weight";

    const auto* w_hint = weight_tensor_->data()->get_bind_hint();
    if(w_hint == nullptr)
    {
        throw std::runtime_error(
            "Linear::export_hf: weight has no bind_hint data");
    }

    // NNTile column-major (input_dim, output_dim) has the same byte layout
    // as HF row-major (output_dim, input_dim). Just write with HF shape.
    writer.add_tensor(
        w_name, dtype_,
        {static_cast<std::int64_t>(output_dim_),
         static_cast<std::int64_t>(input_dim_)},
        *w_hint);

    // Bias: 1D, no conversion
    if(bias_tensor_ != nullptr)
    {
        const auto* b_hint = bias_tensor_->data()->get_bind_hint();
        if(b_hint != nullptr)
        {
            writer.add_tensor(
                hf_prefix + ".bias", dtype_,
                {static_cast<std::int64_t>(output_dim_)},
                *b_hint);
        }
    }
}

//! Get string representation with dimensions
std::string Linear::repr() const
{
    std::string result = "Linear(in=" + std::to_string(input_dim_) +
                         ", out=" + std::to_string(output_dim_);
    if(has_bias())
    {
        result += ", bias=true";
    }
    result += ")";
    return result;
}

} // namespace nntile::module
