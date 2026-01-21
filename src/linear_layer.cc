/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/linear_layer.cc
 * Linear layer implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/linear_layer.hh"

// Include NNTile headers

namespace nntile
{

//! Constructor: stores graph reference, name and dimensions
LinearLayer::LinearLayer(graph::LogicalGraph& graph, const std::string& name, Index input_dim, Index output_dim,
                         graph::DataType dtype)
    : graph_(graph)
    , name_(name)
    , weight_tensor_(nullptr)
    , input_tensor_(nullptr)
    , output_tensor_(nullptr)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , dtype_(dtype)
{
    // Create weight tensor during construction
    weight_tensor_ = &graph_.tensor(
        graph::TensorSpec({input_dim_, output_dim_}, dtype_), weight_name());
}

//! Build forward operation and return output tensor
graph::TensorNode& LinearLayer::build_forward(graph::TensorNode& input_tensor)
{
    // Verify input tensor has correct feature dimension
    if (input_tensor.shape().back() != input_dim_) {
        throw std::invalid_argument("Input tensor feature dimension mismatch in LinearLayer::build_forward");
    }

    // Store reference to input tensor
    input_tensor_ = &input_tensor;

    // Create output tensor with same leading dimensions but output_dim features
    std::vector<Index> output_shape = input_tensor.shape();
    output_shape.back() = output_dim_;
    output_tensor_ = &graph_.tensor(
        graph::TensorSpec(output_shape, dtype_), output_name());

    // Linear transformation: output = input @ weight
    // Uses ndim=1, batch_ndim=0 for arbitrary dimensional tensors
    graph_.add_op(
        graph::OpType::GEMM,
        graph::OpAttrs{graph::GemmAttrs{false, false, 1.0, 0.0, 1, 0}},
        {&input_tensor, weight_tensor_},
        {output_tensor_}
    );

    return *output_tensor_;
}

//! Build backward operation
void LinearLayer::build_backward(graph::TensorNode& grad_output_tensor, graph::TensorNode& grad_input_tensor)
{
    // Verify tensor dimensions
    if (grad_output_tensor.shape().back() != output_dim_) {
        throw std::invalid_argument("Grad output tensor feature dimension mismatch in LinearLayer::build_backward");
    }
    if (grad_input_tensor.shape().back() != input_dim_) {
        throw std::invalid_argument("Grad input tensor feature dimension mismatch in LinearLayer::build_backward");
    }
    if (grad_output_tensor.shape() != grad_input_tensor.shape()) {
        // Leading dimensions should match (batch dimensions)
        bool shape_match = true;
        for (size_t i = 0; i < grad_output_tensor.shape().size() - 1; ++i) {
            if (grad_output_tensor.shape()[i] != grad_input_tensor.shape()[i]) {
                shape_match = false;
                break;
            }
        }
        if (!shape_match) {
            throw std::invalid_argument("Grad output and input tensor shapes don't match in LinearLayer::build_backward");
        }
    }

    // Check that build_forward was called
    if (!input_tensor_ || !output_tensor_) {
        throw std::runtime_error("Forward tensors not initialized - call build_forward first");
    }

    // Create weight gradient tensor
    auto& grad_weight_tensor = graph_.tensor(
        graph::TensorSpec({input_dim_, output_dim_}, dtype_), grad_weight_name());

    // Backward operations:
    // grad_weight = input^T @ grad_output (accumulate weight gradients)
    graph_.add_op(
        graph::OpType::GEMM,
        graph::OpAttrs{graph::GemmAttrs{true, false, 1.0, 0.0, 1, 0}},
        {input_tensor_, &grad_output_tensor},
        {&grad_weight_tensor}
    );

    // grad_input = grad_output @ weight^T (compute input gradients, accumulate into grad_input_tensor)
    graph_.add_op(
        graph::OpType::GEMM,
        graph::OpAttrs{graph::GemmAttrs{false, true, 1.0, 0.0, 1, 0}},
        {&grad_output_tensor, weight_tensor_},
        {&grad_input_tensor}
    );
}

} // namespace nntile