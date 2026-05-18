/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/embedding_backward.hh
 * TileGraph embedding_backward operation: vocab += scatter(embed, index)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Embedding backward: vocab += scatter(embed, index)
struct TileEmbeddingBackwardOp : TileGraph::OpNode
{
    Index m = 0, n = 0, k = 0, k_start = 0, k_size = 0;
    int redux = 0;
    TileGraph::TileNode* index = nullptr, * embed = nullptr, * vocab = nullptr;
    TileEmbeddingBackwardOp() = default;
    TileEmbeddingBackwardOp(Index a, Index b, Index c, Index ks, Index kz, TileGraph::TileNode* i, TileGraph::TileNode* e, TileGraph::TileNode* v, int r = 0) : m(a), n(b), k(c), k_start(ks), k_size(kz), redux(r), index(i), embed(e), vocab(v)
    {
        inputs_ = {index, embed, vocab};
        outputs_ = {vocab};
    }
    std::string op_name() const override { return "TILE_EMBEDDING_BACKWARD"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileEmbeddingBackwardOp>(*this);
    }
};
//! Embedding backward: vocab += scatter(embed, index)
void embedding_backward(
    Index m, Index n, Index k, Index k_start, Index k_size, TileGraph::TileNode* index, TileGraph::TileNode* embed, TileGraph::TileNode* vocab, int redux = 0);
} // namespace nntile::graph::tile_graph
