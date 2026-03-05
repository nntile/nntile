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
#include <stdexcept>

namespace nntile::module
{

//! Constructor: creates new vocab tensor
Embedding::Embedding(graph::NNGraph& graph,
                     const std::string& name,
                     Index num_embeddings,
                     Index embed_dim,
                     graph::DataType dtype)
    : ModuleBase(graph, name)
    , num_embeddings_(num_embeddings)
    , embed_dim_(embed_dim)
    , axis_(-1)
    , redux_(0)
    , dtype_(dtype)
{
    // Create vocab tensor: [embed_dim, num_embeddings] (NNTile layout, transpose of PyTorch)
    vocab_tensor_ = graph_.tensor(
        {embed_dim_, num_embeddings_},
        tensor_name("vocab"),
        dtype_,
        true);
    register_parameter("vocab", vocab_tensor_);
}

//! Constructor: creates new vocab tensor with custom axis and redux
Embedding::Embedding(graph::NNGraph& graph,
                     const std::string& name,
                     Index num_embeddings,
                     Index embed_dim,
                     Index axis,
                     int redux,
                     graph::DataType dtype)
    : ModuleBase(graph, name)
    , num_embeddings_(num_embeddings)
    , embed_dim_(embed_dim)
    , axis_(axis)
    , redux_(redux)
    , dtype_(dtype)
{
    vocab_tensor_ = graph_.tensor(
        {embed_dim_, num_embeddings_},
        tensor_name("vocab"),
        dtype_,
        true);
    register_parameter("vocab", vocab_tensor_);
}

//! Constructor: uses existing vocab tensor
Embedding::Embedding(graph::NNGraph& graph,
                     const std::string& name,
                     graph::NNGraph::TensorNode& vocab_tensor)
    : ModuleBase(graph, name)
    , vocab_tensor_(&vocab_tensor)
    , num_embeddings_(0)
    , embed_dim_(0)
    , axis_(-1)
    , redux_(0)
    , dtype_(vocab_tensor.dtype())
{
    if(vocab_tensor.ndim() != 2)
    {
        throw std::invalid_argument(
            "Embedding::Embedding: vocab tensor must have 2 dimensions, "
            "got " + std::to_string(vocab_tensor.ndim()));
    }

    const auto& v_shape = vocab_tensor.shape();
    embed_dim_ = v_shape[0];
    num_embeddings_ = v_shape[1];

    register_parameter("vocab", vocab_tensor_);
}

//! Constructor: uses existing vocab tensor with custom axis and redux
Embedding::Embedding(graph::NNGraph& graph,
                     const std::string& name,
                     graph::NNGraph::TensorNode& vocab_tensor,
                     Index axis,
                     int redux)
    : ModuleBase(graph, name)
    , vocab_tensor_(&vocab_tensor)
    , num_embeddings_(0)
    , embed_dim_(0)
    , axis_(axis)
    , redux_(redux)
    , dtype_(vocab_tensor.dtype())
{
    if(vocab_tensor.ndim() != 2)
    {
        throw std::invalid_argument(
            "Embedding::Embedding: vocab tensor must have 2 dimensions, "
            "got " + std::to_string(vocab_tensor.ndim()));
    }

    const auto& v_shape = vocab_tensor.shape();
    embed_dim_ = v_shape[0];
    num_embeddings_ = v_shape[1];

    register_parameter("vocab", vocab_tensor_);
}

graph::NNGraph::TensorNode& Embedding::build_forward(
    graph::NNGraph::TensorNode& index)
{
    if(index.ndim() < 1)
    {
        throw std::invalid_argument(
            "Embedding::build_forward: index tensor must have at least one "
            "dimension, got 0-dimensional (scalar) tensor");
    }
    if(index.dtype() != graph::DataType::INT64)
    {
        throw std::invalid_argument(
            "Embedding::build_forward: index tensor must have INT64 dtype");
    }

    index_tensor_ = &index;

    // Use index.ndim() as axis when axis_ < 0 (default "append" behavior)
    Index use_axis = (axis_ < 0) ? index.ndim() : axis_;

    output_tensor_ = graph::embedding(
        &index,
        vocab_tensor_,
        tensor_name("output"),
        use_axis,
        redux_);

    return *output_tensor_;
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
