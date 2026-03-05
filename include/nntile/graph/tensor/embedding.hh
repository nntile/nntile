/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/embedding.hh
 * TensorGraph embedding operation: embed = vocab[index]
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Embedding operation: embed = vocab[index]
struct TensorEmbeddingOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* index = nullptr;
    TensorGraph::TensorNode* vocab = nullptr;
    TensorGraph::TensorNode* embed = nullptr;
    Index axis;

    TensorEmbeddingOp() = default;
    TensorEmbeddingOp(TensorGraph::TensorNode* index_,
                     TensorGraph::TensorNode* vocab_,
                     TensorGraph::TensorNode* embed_,
                     Index axis_)
        : index(index_), vocab(vocab_), embed(embed_), axis(axis_)
    {
        inputs_ = {index, vocab};
        outputs_ = {embed};
    }

    std::string op_name() const override { return "EMBEDDING"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorEmbeddingOp>(*this);
    }
};

//! Embedding: embed = vocab[index]
TensorGraph::TensorNode* embedding(
    TensorGraph::TensorNode* index,
    TensorGraph::TensorNode* vocab,
    const std::string& output_name,
    Index axis);

void embedding(TensorGraph::TensorNode* index,
               TensorGraph::TensorNode* vocab,
               TensorGraph::TensorNode* embed,
               Index axis);

} // namespace nntile::graph::tensor
