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
    weight_tensor_ = &graph_.tensor(
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
    weight_tensor_ = &graph_.tensor(
        {input_dim_, output_dim_},
        tensor_name("weight"),
        dtype_,
        true);
    register_parameter("weight", weight_tensor_);

    // Create bias tensor if requested
    if(with_bias)
    {
        bias_tensor_ = &graph_.tensor(
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

    // Create output tensor with same leading dimensions but output_dim features
    std::vector<Index> output_shape = input.shape();
    output_shape.back() = output_dim_;
    bool output_requires_grad = graph_.requires_grad(input) ||
        graph_.requires_grad(*weight_tensor_);
    if(bias_tensor_ != nullptr)
    {
        output_requires_grad = output_requires_grad ||
            graph_.requires_grad(*bias_tensor_);
    }

    output_tensor_ = &graph_.tensor(
        std::move(output_shape),
        tensor_name("output"),
        dtype_,
        output_requires_grad);

    // Linear transformation: output = input @ weight
    // Uses ndim=1, batch_ndim=0 for arbitrary dimensional tensors
    graph_.add_op(
        graph::OpType::GEMM,
        graph::OpAttrs{graph::GemmAttrs{false, false, 1.0, 0.0, 1, 0}},
        {&input, weight_tensor_},
        {output_tensor_}
    );

    // Add bias if present: output += bias (broadcast along last dim)
    if(bias_tensor_ != nullptr)
    {
        graph_.add_op(
            graph::OpType::ADD_FIBER,
            graph::OpAttrs{graph::AddFiberAttrs{output_tensor_->ndim() - 1, 0, 1.0, 1.0}},
            {output_tensor_, bias_tensor_},
            {output_tensor_}  // In-place addition
        );
    }

    forward_built_ = true;
    return *output_tensor_;
}

//! Build backward operations using gradient tracking
void Linear::build_backward()
{
    // Check that build_forward was called
    if(!forward_built_ || !input_tensor_ || !output_tensor_)
    {
        throw std::runtime_error(
            "Linear::build_backward: forward not built - "
            "call build_forward first");
    }

    // Get gradient of output tensor (must exist - set by downstream module)
    graph::NNGraph::TensorNode* grad_output = output_tensor_->grad();
    if(!grad_output)
    {
        throw std::runtime_error(
            "Linear::build_backward: no gradient registered for output "
            "tensor '" + output_tensor_->name() + "'");
    }

    // Compute weight gradient if required
    if(graph_.requires_grad(*weight_tensor_))
    {
        // Check if this is the first contribution (beta=0) or accumulation
        bool first_weight_grad = graph_.is_first_grad(*weight_tensor_);
        graph::NNGraph::TensorNode& grad_weight = graph_.get_or_create_grad(
            *weight_tensor_, grad_name("weight"));

        Scalar beta_weight = first_weight_grad ? 0.0 : 1.0;

        // grad_weight += input^T @ grad_output
        graph_.add_op(
            graph::OpType::GEMM,
            graph::OpAttrs{graph::GemmAttrs{true, false, 1.0, beta_weight, 1, 0}},
            {input_tensor_, grad_output},
            {&grad_weight}
        );
    }

    // Compute bias gradient if bias is present
    // grad_bias = sum(grad_output) along all batch dimensions
    if(bias_tensor_ != nullptr && graph_.requires_grad(*bias_tensor_))
    {
        bool first_bias_grad = graph_.is_first_grad(*bias_tensor_);
        graph::NNGraph::TensorNode& grad_bias = graph_.get_or_create_grad(
            *bias_tensor_, grad_name("bias"));

        Scalar beta_bias = first_bias_grad ? 0.0 : 1.0;

        // grad_bias += sum(grad_output, axis=0...-1)
        graph_.add_op(
            graph::OpType::SUM_FIBER,
            graph::OpAttrs{graph::SumFiberAttrs{1.0, beta_bias}},
            {grad_output},
            {&grad_bias}
        );
    }

    // Compute input gradient only if required
    if(graph_.requires_grad(*input_tensor_))
    {
        bool first_input_grad = graph_.is_first_grad(*input_tensor_);
        // Use input tensor's own name for its gradient (input is not a parameter)
        graph::NNGraph::TensorNode& grad_input = graph_.get_or_create_grad(
            *input_tensor_, input_tensor_->name() + "_grad");

        Scalar beta_input = first_input_grad ? 0.0 : 1.0;

        // grad_input += grad_output @ weight^T
        graph_.add_op(
            graph::OpType::GEMM,
            graph::OpAttrs{graph::GemmAttrs{false, true, 1.0, beta_input, 1, 0}},
            {grad_output, weight_tensor_},
            {&grad_input}
        );
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
