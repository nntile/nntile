/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/linear.hh
 * Linear module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NNTILE_HAVE_TORCH
#   include <torch/torch.h>
#endif

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Linear module using graph API
//! Adds linear transformation operations to logical graphs
//!
//! Computes: output = input @ weight + bias (bias optional)
//!
//! Supports flexible construction modes:
//! 1. Create new weight/bias tensors (specify dimensions)
//! 2. Use existing weight/bias tensors (for weight/bias sharing)
class Linear : public Module
{
private:
    // References to parameter tensors (also registered via register_parameter)
    graph::NNGraph::TensorNode* weight_tensor_ = nullptr;
    graph::NNGraph::TensorNode* bias_tensor_ = nullptr;

    graph::NNGraph::TensorNode* input_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

    // Dimensions and data type
    Index input_dim_;
    Index output_dim_;
    graph::DataType dtype_;

public:
    //! Constructor: creates new weight tensor, no bias
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param dtype Data type for tensors
    Linear(
        graph::NNGraph* graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: creates new weight and optionally bias tensors
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param input_dim Input feature dimension
    //! @param output_dim Output feature dimension
    //! @param with_bias Whether to create bias tensor
    //! @param dtype Data type for tensors
    Linear(
        graph::NNGraph* graph,
        const std::string& name,
        Index input_dim,
        Index output_dim,
        bool with_bias,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: uses existing weight tensor, no bias
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param weight_tensor Existing weight tensor to use [input_dim, output_dim]
    Linear(
        graph::NNGraph* graph,
        const std::string& name,
        graph::NNGraph::TensorNode* weight_tensor
    );

    //! Constructor: uses existing weight and bias tensors
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param weight_tensor Existing weight tensor [input_dim, output_dim]
    //! @param bias_tensor Existing bias tensor [output_dim]
    Linear(
        graph::NNGraph* graph,
        const std::string& name,
        graph::NNGraph::TensorNode* weight_tensor,
        graph::NNGraph::TensorNode* bias_tensor
    );

#ifdef NNTILE_HAVE_TORCH
    //! Constructor: creates Linear from torch::nn::Linear (same dimensions)
    //! and binds weight/bias data from the PyTorch layer.
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param linear_layer PyTorch Linear layer to mirror (weight/bias copied)
    //! @param dtype Data type for tensors
    Linear(
        graph::NNGraph* graph,
        const std::string& name,
        const torch::nn::Linear& linear_layer,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Get weight data in NNTile format for runtime.bind_data().
    //! Converts PyTorch [out,in] row-major to NNTile [in,out] column-major.
    static std::vector<float> weight_data_from_pytorch(const torch::Tensor& w);

    //! Get bias data in NNTile format for runtime.bind_data().
    //! Copies 1D tensor [output_dim].
    static std::vector<float> bias_data_from_pytorch(const torch::Tensor& b);
#endif

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* input);

    //! Bind weight data for Runtime::compile(). Data must be in NNTile layout.
    //! Moves data into the graph; call std::move() to avoid copy.
    void bind_weight(std::vector<std::uint8_t> data);

    //! Bind weight data (FP32 convenience; copies into internal buffer).
    void bind_weight(const std::vector<float>& data);

    //! Bind bias data for Runtime::compile(). Data must be in NNTile layout.
    //! Moves data into the graph; call std::move() to avoid copy.
    void bind_bias(std::vector<std::uint8_t> data);

    //! Bind bias data (FP32 convenience; copies into internal buffer).
    void bind_bias(const std::vector<float>& data);

    //! Forward: calls forward (user does bookkeeping via autograd ops)
    graph::NNGraph::TensorNode* operator()(graph::NNGraph::TensorNode* input)
    {
        return forward(input);
    }

    //! Get string representation with dimensions
    std::string repr() const override;

    // Tensor accessors
    graph::NNGraph::TensorNode* weight_tensor() const { return weight_tensor_; }
    graph::NNGraph::TensorNode* bias_tensor() const { return bias_tensor_; }

    // Check if bias is enabled
    bool has_bias() const { return bias_tensor_ != nullptr; }

    // Dimension accessors
    Index input_dim() const { return input_dim_; }
    Index output_dim() const { return output_dim_; }
    graph::DataType dtype() const { return dtype_; }
};

#ifdef NNTILE_HAVE_TORCH

inline Linear::Linear(graph::NNGraph* graph,
                     const std::string& name,
                     const torch::nn::Linear& linear_layer,
                     graph::DataType dtype)
    : Linear(graph, name,
             static_cast<Index>(linear_layer->weight.size(1)),
             static_cast<Index>(linear_layer->weight.size(0)),
             linear_layer->options.bias(),
             dtype)
{
    bind_weight(weight_data_from_pytorch(linear_layer->weight));
    if(linear_layer->options.bias())
    {
        bind_bias(bias_data_from_pytorch(linear_layer->bias));
    }
}

inline std::vector<float> Linear::weight_data_from_pytorch(
    const torch::Tensor& w)
{
    if(!w.defined())
    {
        throw std::invalid_argument(
            "Linear::weight_data_from_pytorch: tensor undefined");
    }
    if(w.dim() != 2)
    {
        throw std::invalid_argument(
            "Linear::weight_data_from_pytorch: expected 2D tensor");
    }
    const long out = w.size(0);
    const long in = w.size(1);
    std::vector<float> result(static_cast<size_t>(in * out));
    auto acc = w.accessor<float, 2>();
    for(long j = 0; j < out; ++j)
    {
        for(long i = 0; i < in; ++i)
        {
            result[static_cast<size_t>(i + j * in)] = acc[j][i];
        }
    }
    return result;
}

inline std::vector<float> Linear::bias_data_from_pytorch(
    const torch::Tensor& b)
{
    if(!b.defined())
    {
        throw std::invalid_argument(
            "Linear::bias_data_from_pytorch: tensor undefined");
    }
    if(b.dim() != 1)
    {
        throw std::invalid_argument(
            "Linear::bias_data_from_pytorch: expected 1D tensor");
    }
    const long n = b.size(0);
    std::vector<float> result(static_cast<size_t>(n));
    auto acc = b.accessor<float, 1>();
    for(long i = 0; i < n; ++i)
    {
        result[static_cast<size_t>(i)] = acc[i];
    }
    return result;
}

#endif // NNTILE_HAVE_TORCH

} // namespace nntile::module
