/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/module/embedding.hh
 * Embedding module implementation using NNTile graph API.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include NNTile headers
#include <nntile/graph.hh>
#include <nntile/module/module.hh>

namespace nntile::module
{

//! Embedding module using graph API
//! Adds embedding lookup operations to logical graphs
//!
//! Computes: output = vocab[index]
//! Index tensor (INT64) selects rows from vocab; output shape is index.shape + [embed_dim]
//!
//! Supports flexible construction modes:
//! 1. Create new vocab tensor (specify num_embeddings, embed_dim)
//! 2. Use existing vocab tensor (for weight sharing)
class Embedding : public Module
{
private:
    // Reference to parameter tensor (also registered via register_parameter)
    graph::NNGraph::TensorNode* vocab_tensor_ = nullptr;

    graph::NNGraph::TensorNode* index_tensor_ = nullptr;
    graph::NNGraph::TensorNode* output_tensor_ = nullptr;

    // Dimensions and data type
    Index num_embeddings_;
    Index embed_dim_;
    Index axis_;
    int redux_;
    graph::DataType dtype_;

public:
    //! Constructor: creates new vocab tensor
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param num_embeddings Size of the vocabulary
    //! @param embed_dim Size of each embedding vector
    //! @param dtype Data type for tensors
    Embedding(
        graph::NNGraph& graph,
        const std::string& name,
        Index num_embeddings,
        Index embed_dim,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: creates new vocab tensor with custom axis and redux
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param num_embeddings Size of the vocabulary
    //! @param embed_dim Size of each embedding vector
    //! @param axis Axis along which embedding dimension is inserted (default: append)
    //! @param redux Reduction mode for backward (0=no reduction, 1=reduce)
    //! @param dtype Data type for tensors
    Embedding(
        graph::NNGraph& graph,
        const std::string& name,
        Index num_embeddings,
        Index embed_dim,
        Index axis,
        int redux,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: uses existing vocab tensor
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param vocab_tensor Existing vocab tensor [num_embeddings, embed_dim]
    Embedding(
        graph::NNGraph& graph,
        const std::string& name,
        graph::NNGraph::TensorNode& vocab_tensor
    );

    //! Constructor: uses existing vocab tensor with custom axis and redux
    //! @param graph The neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param vocab_tensor Existing vocab tensor [num_embeddings, embed_dim]
    //! @param axis Axis along which embedding dimension is inserted
    //! @param redux Reduction mode for backward
    Embedding(
        graph::NNGraph& graph,
        const std::string& name,
        graph::NNGraph::TensorNode& vocab_tensor,
        Index axis,
        int redux
    );

    graph::NNGraph::TensorNode& build_forward(
        graph::NNGraph::TensorNode& index);

    //! Forward: calls build_forward (user does bookkeeping via autograd ops)
    graph::NNGraph::TensorNode& operator()(graph::NNGraph::TensorNode& index)
    {
        return build_forward(index);
    }

    //! Get string representation with dimensions
    std::string repr() const override;

    // Tensor accessors
    graph::NNGraph::TensorNode* vocab_tensor() const { return vocab_tensor_; }

    // Dimension accessors
    Index num_embeddings() const { return num_embeddings_; }
    Index embed_dim() const { return embed_dim_; }
    Index axis() const { return axis_; }
    int redux() const { return redux_; }
    graph::DataType dtype() const { return dtype_; }
};

} // namespace nntile::module
