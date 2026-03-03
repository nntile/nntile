/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/embedding_backward.hh
 * TensorGraph embedding_backward operation: vocab += scatter(embed, index)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Embedding backward: vocab += scatter(embed, index)
struct TensorEmbeddingBackwardOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* index = nullptr;
    TensorGraph::TensorNode* embed = nullptr;
    TensorGraph::TensorNode* vocab = nullptr;
    Index axis = 0;
    int redux = 0;

    TensorEmbeddingBackwardOp() = default;
    TensorEmbeddingBackwardOp(TensorGraph::TensorNode* index_,
                             TensorGraph::TensorNode* embed_,
                             TensorGraph::TensorNode* vocab_,
                             Index axis_,
                             int redux_ = 0)
        : index(index_), embed(embed_), vocab(vocab_), axis(axis_), redux(redux_)
    {
        inputs_ = {index, embed, vocab};
        outputs_ = {vocab};
    }

    std::string op_name() const override { return "EMBEDDING_BACKWARD"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorEmbeddingBackwardOp>(*this);
    }
};

//! Embedding backward: vocab += scatter(embed, index)
void embedding_backward(TensorGraph::TensorNode* index,
                        TensorGraph::TensorNode* embed,
                        TensorGraph::TensorNode* vocab,
                        Index axis,
                        int redux = 0);

} // namespace nntile::graph
