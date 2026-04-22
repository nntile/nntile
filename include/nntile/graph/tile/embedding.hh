/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/embedding.hh
 * TileGraph embedding operation: embed = vocab[index]
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tile/graph.hh>

namespace nntile::graph::tile_graph
{

//! Embedding operation: embed = vocab[index]
struct TileEmbeddingOp : TileGraph::OpNode
{
    Index m = 0, n = 0, k = 0, k_start = 0, k_size = 0;
    TileGraph::TileNode* index = nullptr; // int64
    TileGraph::TileNode* vocab = nullptr;
    TileGraph::TileNode* embed = nullptr;
    TileEmbeddingOp() = default;
    TileEmbeddingOp(Index a, Index b, Index c, Index ks, Index kz, TileGraph::TileNode* i, TileGraph::TileNode* v, TileGraph::TileNode* e) : m(a), n(b), k(c), k_start(ks), k_size(kz), index(i), vocab(v), embed(e)
    {
        inputs_ = {index, vocab};
        outputs_ = {embed};
    }
    std::string op_name() const override { return "TILE_EMBEDDING"; }
    void execute(TileGraph::Runtime& runtime) const override;
    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileEmbeddingOp>(*this);
    }
};
void embedding(Index m, Index n, Index k, Index k_start, Index k_size, TileGraph::TileNode* index, TileGraph::TileNode* vocab, TileGraph::TileNode* embed);
} // namespace nntile::graph::tile_graph
