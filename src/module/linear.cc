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
#include <stdexcept>

namespace nntile::module
{

namespace
{

// GEMM scaling: use full weight of matrix product
constexpr Scalar GEMM_ALPHA = 1.0;
// GEMM: overwrite output (don't accumulate into existing)
constexpr Scalar GEMM_BETA_OVERWRITE = 0.0;
// GEMM: accumulate into output
constexpr Scalar GEMM_BETA_ACCUMULATE = 1.0;
// Matrix multiply: one contraction dimension (K)
constexpr Index GEMM_NDIM_MATRIX = 1;
// No batch dimensions
constexpr Index NO_BATCH_DIM = 0;
// Operand transpose flags
constexpr bool NO_TRANSPOSE = false;
constexpr bool TRANSPOSE = true;

// Add fiber: use full weight of fiber, add to existing tensor
constexpr Scalar ADD_FIBER_ALPHA = 1.0;
constexpr Scalar ADD_FIBER_BETA = 1.0;

// Sum fiber: no batch dims, no redux
constexpr Index SUM_FIBER_BATCH_NDIM = 0;
constexpr int SUM_FIBER_REDUX_NONE = 0;
constexpr Scalar SUM_FIBER_ALPHA = 1.0;
constexpr Scalar SUM_FIBER_BETA_OVERWRITE = 0.0;
constexpr Scalar SUM_FIBER_BETA_ACCUMULATE = 1.0;

} // anonymous namespace

//! Constructor: creates new weight tensor, no bias
Linear::Linear(graph::NNGraph& graph,
               const std::string& name,
               Index input_dim, Index output_dim,
               graph::DataType dtype)
    : Module(graph, name)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Create weight tensor during construction
    weight_tensor_ = graph_.tensor(
        {input_dim_, output_dim_},
        tensor_name("weight"),
        dtype_,
        true);
    register_parameter("weight", weight_tensor_);
}

//! Constructor: creates new weight and optionally bias tensors
Linear::Linear(graph::NNGraph& graph,
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
    weight_tensor_ = graph_.tensor(
        {input_dim_, output_dim_},
        tensor_name("weight"),
        dtype_,
        true);
    register_parameter("weight", weight_tensor_);

    // Create bias tensor if requested
    if(with_bias)
    {
        bias_tensor_ = graph_.tensor(
            {output_dim_},
            tensor_name("bias"),
            dtype_,
            true);
        register_parameter("bias", bias_tensor_);
    }
}

//! Constructor: uses existing weight tensor, no bias
Linear::Linear(graph::NNGraph& graph,
               const std::string& name,
               graph::NNGraph::TensorNode& weight_tensor)
    : Module(graph, name)
    , weight_tensor_(&weight_tensor)
    , input_dim_(0)
    , output_dim_(0)
    , dtype_(weight_tensor.dtype())
{
    // Validate weight tensor has at least 2 dimensions
    if(weight_tensor.ndim() < 2)
    {
        throw std::invalid_argument(
            "Linear::Linear: weight tensor must have at least 2 dimensions, "
            "got " + std::to_string(weight_tensor.ndim()));
    }

    // Extract dimensions from weight tensor shape
    // Weight shape is [input_dim, output_dim]
    const auto& w_shape = weight_tensor.shape();
    input_dim_ = w_shape[0];
    output_dim_ = w_shape[1];

    register_parameter("weight", weight_tensor_);
}

//! Constructor: uses existing weight and bias tensors
Linear::Linear(graph::NNGraph& graph,
               const std::string& name,
               graph::NNGraph::TensorNode& weight_tensor,
               graph::NNGraph::TensorNode& bias_tensor)
    : Module(graph, name)
    , weight_tensor_(&weight_tensor)
    , bias_tensor_(&bias_tensor)
    , input_dim_(0)
    , output_dim_(0)
    , dtype_(weight_tensor.dtype())
{
    // Validate weight tensor has at least 2 dimensions
    if(weight_tensor.ndim() < 2)
    {
        throw std::invalid_argument(
            "Linear::Linear: weight tensor must have at least 2 dimensions, "
            "got " + std::to_string(weight_tensor.ndim()));
    }

    // Extract dimensions from weight tensor shape
    // Weight shape is [input_dim, output_dim]
    const auto& w_shape = weight_tensor.shape();
    input_dim_ = w_shape[0];
    output_dim_ = w_shape[1];

    // Validate bias tensor shape
    if(bias_tensor.ndim() != 1)
    {
        throw std::invalid_argument(
            "Linear::Linear: bias tensor must be 1-dimensional, "
            "got " + std::to_string(bias_tensor.ndim()));
    }
    if(bias_tensor.shape()[0] != output_dim_)
    {
        throw std::invalid_argument(
            "Linear::Linear: bias tensor dimension mismatch. "
            "Expected " + std::to_string(output_dim_) + ", got " +
            std::to_string(bias_tensor.shape()[0]));
    }

    // Validate data types match
    if(bias_tensor.dtype() != dtype_)
    {
        throw std::invalid_argument(
            "Linear::Linear: bias tensor dtype must match weight tensor dtype");
    }

    register_parameter("weight", weight_tensor_);
    register_parameter("bias", bias_tensor_);
}

//! Build forward operation and return output tensor
graph::NNGraph::TensorNode& Linear::build_forward(graph::NNGraph::TensorNode& input)
{
    // Verify input tensor has at least one dimension (avoid undefined behavior
    // when calling .back() on empty shape for 0-dimensional scalar tensors)
    if(input.ndim() < 1)
    {
        throw std::invalid_argument(
            "Linear::build_forward: input tensor must have at least one "
            "dimension, got 0-dimensional (scalar) tensor");
    }

    // Verify input tensor has correct feature dimension
    if(input.shape().back() != input_dim_)
    {
        throw std::invalid_argument(
            "Linear::build_forward: input tensor feature dimension mismatch. "
            "Expected " + std::to_string(input_dim_) + ", got " +
            std::to_string(input.shape().back()));
    }

    // Store reference to input tensor
    input_tensor_ = &input;

    // Linear transformation (autograd Gemm)
    const std::string gemm_name = bias_tensor_ != nullptr
        ? tensor_name("gemm_output")
        : tensor_name("output");
    graph::NNGraph::TensorNode* gemm_out = graph::gemm(
        &input,
        weight_tensor_,
        gemm_name,
        GEMM_ALPHA,
        NO_TRANSPOSE,
        NO_TRANSPOSE,
        GEMM_NDIM_MATRIX,
        NO_BATCH_DIM);

    // Add bias if present (autograd AddFiber)
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

    return *output_tensor_;
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
