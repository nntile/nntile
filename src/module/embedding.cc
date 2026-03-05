/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/module/embedding.cc
 * Embedding module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/module/embedding.hh"

// Include standard headers
#include <cstring>
#include <stdexcept>

namespace nntile::module
{

//! Constructor: creates new vocab tensor
Embedding::Embedding(graph::NNGraph* graph,
                     const std::string& name,
                     Index num_embeddings,
                     Index embed_dim,
                     graph::DataType dtype)
    : Module(graph, name)
    , num_embeddings_(num_embeddings)
    , embed_dim_(embed_dim)
    , axis_(-1)
    , redux_(0)
    , dtype_(dtype)
{
    // Create vocab tensor: [embed_dim, num_embeddings] (NNTile layout, transpose of PyTorch)
    vocab_tensor_ = graph_->tensor(
        {embed_dim_, num_embeddings_},
        tensor_name("vocab"),
        dtype_,
        true);
    register_parameter("vocab", vocab_tensor_);
}

//! Constructor: creates new vocab tensor with custom axis and redux
Embedding::Embedding(graph::NNGraph* graph,
                     const std::string& name,
                     Index num_embeddings,
                     Index embed_dim,
                     Index axis,
                     int redux,
                     graph::DataType dtype)
    : Module(graph, name)
    , num_embeddings_(num_embeddings)
    , embed_dim_(embed_dim)
    , axis_(axis)
    , redux_(redux)
    , dtype_(dtype)
{
    vocab_tensor_ = graph_->tensor(
        {embed_dim_, num_embeddings_},
        tensor_name("vocab"),
        dtype_,
        true);
    register_parameter("vocab", vocab_tensor_);
}

//! Constructor: uses existing vocab tensor
Embedding::Embedding(graph::NNGraph* graph,
                     const std::string& name,
                     graph::NNGraph::TensorNode* vocab_tensor)
    : Module(graph, name)
    , vocab_tensor_(vocab_tensor)
    , num_embeddings_(0)
    , embed_dim_(0)
    , axis_(-1)
    , redux_(0)
    , dtype_(vocab_tensor != nullptr ? vocab_tensor->dtype() : graph::DataType::FP32)
{
    if(vocab_tensor == nullptr)
    {
        throw std::invalid_argument(
            "Embedding::Embedding: vocab_tensor must be non-null");
    }
    if(vocab_tensor->ndim() != 2)
    {
        throw std::invalid_argument(
            "Embedding::Embedding: vocab tensor must have 2 dimensions, "
            "got " + std::to_string(vocab_tensor->ndim()));
    }

    const auto& v_shape = vocab_tensor->shape();
    embed_dim_ = v_shape[0];
    num_embeddings_ = v_shape[1];

    register_parameter("vocab", vocab_tensor_);
}

//! Constructor: uses existing vocab tensor with custom axis and redux
Embedding::Embedding(graph::NNGraph* graph,
                     const std::string& name,
                     graph::NNGraph::TensorNode* vocab_tensor,
                     Index axis,
                     int redux)
    : Module(graph, name)
    , vocab_tensor_(vocab_tensor)
    , num_embeddings_(0)
    , embed_dim_(0)
    , axis_(axis)
    , redux_(redux)
    , dtype_(vocab_tensor != nullptr ? vocab_tensor->dtype() : graph::DataType::FP32)
{
    if(vocab_tensor == nullptr)
    {
        throw std::invalid_argument(
            "Embedding::Embedding: vocab_tensor must be non-null");
    }
    if(vocab_tensor->ndim() != 2)
    {
        throw std::invalid_argument(
            "Embedding::Embedding: vocab tensor must have 2 dimensions, "
            "got " + std::to_string(vocab_tensor->ndim()));
    }

    const auto& v_shape = vocab_tensor->shape();
    embed_dim_ = v_shape[0];
    num_embeddings_ = v_shape[1];

    register_parameter("vocab", vocab_tensor_);
}

graph::NNGraph::TensorNode* Embedding::forward(
    graph::NNGraph::TensorNode* index)
{
    if(index == nullptr)
    {
        throw std::invalid_argument(
            "Embedding::forward: index tensor must be non-null");
    }
    if(index->ndim() < 1)
    {
        throw std::invalid_argument(
            "Embedding::forward: index tensor must have at least one "
            "dimension, got 0-dimensional (scalar) tensor");
    }
    if(index->dtype() != graph::DataType::INT64)
    {
        throw std::invalid_argument(
            "Embedding::forward: index tensor must have INT64 dtype");
    }

    index_tensor_ = index;

    // Use index.ndim() as axis when axis_ < 0 (default "append" behavior)
    Index use_axis = (axis_ < 0) ? index->ndim() : axis_;

    output_tensor_ = graph::embedding(
        index,
        vocab_tensor_,
        tensor_name("output"),
        use_axis,
        redux_);

    return output_tensor_;
}

void Embedding::bind_weight(std::vector<std::uint8_t> data)
{
    if(vocab_tensor_ == nullptr)
    {
        throw std::runtime_error(
            "Embedding::bind_weight: vocab tensor is null (external vocab mode)");
    }
    vocab_tensor_->data()->set_bind_hint(std::move(data));
    vocab_tensor_->mark_input(true);
}

void Embedding::bind_weight(const std::vector<float>& data)
{
    const size_t expected =
        static_cast<size_t>(embed_dim_) * num_embeddings_;
    if(data.size() != expected)
    {
        throw std::invalid_argument(
            "Embedding::bind_weight: size mismatch, expected " +
            std::to_string(expected) + " elements, got " +
            std::to_string(data.size()));
    }
    std::vector<std::uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    bind_weight(std::move(bytes));
}

//! Get string representation with dimensions
std::string Embedding::repr() const
{
    std::string result = "Embedding(num_embeddings=" +
                         std::to_string(num_embeddings_) +
                         ", embed_dim=" + std::to_string(embed_dim_) + ")";
    return result;
}

} // namespace nntile::module
