/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/embedding.hh
 * NNGraph embedding autograd operation.
 *
 * Forward: embed = vocab[index]
 * Backward: grad_vocab += embedding_backward(index, grad_embed)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph_op_node.hh>

namespace nntile::graph
{

//! Embedding op: embed = vocab[index]. PyTorch-style: outputs created in forward().
//! index is INT64 (typically no grad); only vocab gets grad.
struct NNEmbeddingOp : NNGraph::OpNode
{
    Index axis;
    int redux;
    NNGraph::TensorNode* index = nullptr;
    NNGraph::TensorNode* vocab = nullptr;

    NNEmbeddingOp() = default;
    NNEmbeddingOp(NNGraph::TensorNode* index_,
                 NNGraph::TensorNode* vocab_,
                 Index axis_ = 0,
                 int redux_ = 0)
        : axis(axis_), redux(redux_), index(index_), vocab(vocab_)
    {
        inputs_ = {index, vocab};
    }

    NNGraph::TensorNode* forward(const std::string& output_name);
    void backward() const override;
};

NNGraph::TensorNode* embedding(
    NNGraph::TensorNode* index,
    NNGraph::TensorNode* vocab,
    const std::string& output_name,
    Index axis = 0,
    int redux = 0);

} // namespace nntile::graph
