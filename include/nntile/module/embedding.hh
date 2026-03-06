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
#include <cstdint>
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
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param num_embeddings Size of the vocabulary
    //! @param embed_dim Size of each embedding vector
    //! @param dtype Data type for tensors
    Embedding(
        graph::NNGraph* graph,
        const std::string& name,
        Index num_embeddings,
        Index embed_dim,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: creates new vocab tensor with custom axis and redux
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param num_embeddings Size of the vocabulary
    //! @param embed_dim Size of each embedding vector
    //! @param axis Axis along which embedding dimension is inserted (default: append)
    //! @param redux Reduction mode for backward (0=no reduction, 1=reduce)
    //! @param dtype Data type for tensors
    Embedding(
        graph::NNGraph* graph,
        const std::string& name,
        Index num_embeddings,
        Index embed_dim,
        Index axis,
        int redux,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Constructor: uses existing vocab tensor
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param vocab_tensor Existing vocab tensor [num_embeddings, embed_dim]
    Embedding(
        graph::NNGraph* graph,
        const std::string& name,
        graph::NNGraph::TensorNode* vocab_tensor
    );

    //! Constructor: uses existing vocab tensor with custom axis and redux
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param vocab_tensor Existing vocab tensor [num_embeddings, embed_dim]
    //! @param axis Axis along which embedding dimension is inserted
    //! @param redux Reduction mode for backward
    Embedding(
        graph::NNGraph* graph,
        const std::string& name,
        graph::NNGraph::TensorNode* vocab_tensor,
        Index axis,
        int redux
    );

#ifdef NNTILE_HAVE_TORCH
    //! Constructor: creates Embedding from torch::nn::Embedding (same dimensions)
    //! and binds vocab data from the PyTorch layer.
    //! @param graph Pointer to the neural network graph this module belongs to
    //! @param name Layer name (used to generate unique tensor names)
    //! @param embedding_layer PyTorch Embedding layer to mirror (weight copied)
    //! @param dtype Data type for tensors
    Embedding(
        graph::NNGraph* graph,
        const std::string& name,
        const torch::nn::Embedding& embedding_layer,
        graph::DataType dtype = graph::DataType::FP32
    );

    //! Get vocab data in NNTile format for runtime.bind_data().
    //! Converts PyTorch [num_embeddings, embed_dim] row-major to NNTile
    //! [embed_dim, num_embeddings] column-major.
    static std::vector<float> vocab_data_from_pytorch(const torch::Tensor& w);
#endif

    graph::NNGraph::TensorNode* forward(
        graph::NNGraph::TensorNode* index);

    //! Bind vocab (weight) data for Runtime::compile(). Data must be in NNTile
    //! layout [embed_dim, num_embeddings] column-major.
    //! Moves data into the graph; call std::move() to avoid copy.
    void bind_weight(std::vector<std::uint8_t> data);

    //! Bind vocab (weight) data (FP32 convenience; copies into internal buffer).
    void bind_weight(const std::vector<float>& data);

    //! Forward: calls forward (user does bookkeeping via autograd ops)
    graph::NNGraph::TensorNode* operator()(graph::NNGraph::TensorNode* index)
    {
        return forward(index);
    }

    //! HF import: read embedding weight from an HF SafeTensors file.
    //! HF stores (num_embeddings, embed_dim) row-major, which is byte-
    //! identical to NNTile (embed_dim, num_embeddings) column-major.
    void import_hf(const io::SafeTensorsReader& reader,
                   const std::string& hf_prefix) override;

    //! HF export: write embedding weight to an HF SafeTensors writer.
    void export_hf(io::SafeTensorsWriter& writer,
                   const std::string& hf_prefix) const override;

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

#ifdef NNTILE_HAVE_TORCH

inline Embedding::Embedding(graph::NNGraph* graph,
                           const std::string& name,
                           const torch::nn::Embedding& embedding_layer,
                           graph::DataType dtype)
    : Embedding(graph, name,
                static_cast<Index>(embedding_layer->weight.size(0)),
                static_cast<Index>(embedding_layer->weight.size(1)),
                dtype)
{
    bind_weight(vocab_data_from_pytorch(embedding_layer->weight));
}

inline std::vector<float> Embedding::vocab_data_from_pytorch(
    const torch::Tensor& w)
{
    if(!w.defined())
    {
        throw std::invalid_argument(
            "Embedding::vocab_data_from_pytorch: tensor undefined");
    }
    if(w.dim() != 2)
    {
        throw std::invalid_argument(
            "Embedding::vocab_data_from_pytorch: expected 2D tensor");
    }
    // PyTorch weight: [num_embeddings, embed_dim]; NNTile vocab: [embed_dim, num_embeddings] col-major
    const long num_emb = w.size(0);
    const long emb_dim = w.size(1);
    std::vector<float> result(static_cast<size_t>(emb_dim * num_emb));
    auto acc = w.accessor<float, 2>();
    for(long j = 0; j < num_emb; ++j)
    {
        for(long i = 0; i < emb_dim; ++i)
        {
            result[static_cast<size_t>(i + j * emb_dim)] = acc[j][i];
        }
    }
    return result;
}

#endif // NNTILE_HAVE_TORCH

} // namespace nntile::module
