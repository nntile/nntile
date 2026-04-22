/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/nn/pytorch_tile_helpers.hh
 * Heterogeneous per-axis tiling helpers for NNGraph PyTorch + TileGraph tests.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_HAVE_TORCH

#   include <vector>

#   include <nntile/graph/nn/graph.hh>

namespace nntile::test
{

//! 2D tensor shape (6, 7): heterogeneous splits on both axes.
inline void nn_pytorch_tile_heterogeneous_rank2_6x7(graph::NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    t->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
}

//! 1D tensor length 7.
inline void nn_pytorch_tile_heterogeneous_1d_len7(graph::NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{3, 4});
}

//! 1D tensor length 6.
inline void nn_pytorch_tile_heterogeneous_1d_len6(graph::NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 4});
}

//! 1D tensor length 4.
inline void nn_pytorch_tile_heterogeneous_1d_len4(graph::NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 2});
}

//! Logits (nclasses, batch) = (5, 7) for cross_entropy.
inline void nn_pytorch_tile_logits_5x7(graph::NNGraph::TensorNode* x)
{
    x->data()->axis(0)->set_tiling(std::vector<Index>{2, 3});
    x->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
}

//! GEMM A (6,7) * B (7,6); K axis set on A only (merged with B axis 0).
inline void nn_pytorch_tile_gemm_operands_6_7_6(
    graph::NNGraph::TensorNode* a, graph::NNGraph::TensorNode* b)
{
    a->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    a->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
    b->data()->axis(1)->set_tiling(std::vector<Index>{2, 4});
}

//! Vocab matrix (embed_dim, num_embeddings) = (10, 10).
inline void nn_pytorch_tile_vocab_10x10(graph::NNGraph::TensorNode* vocab)
{
    vocab->data()->axis(0)->set_tiling(std::vector<Index>{4, 3, 3});
    vocab->data()->axis(1)->set_tiling(std::vector<Index>{4, 6});
}

//! Index tensor (4, 5).
inline void nn_pytorch_tile_index_4x5(graph::NNGraph::TensorNode* index)
{
    index->data()->axis(0)->set_tiling(std::vector<Index>{2, 2});
    index->data()->axis(1)->set_tiling(std::vector<Index>{2, 3});
}

//! Index vector length 3.
inline void nn_pytorch_tile_index_len3(graph::NNGraph::TensorNode* index)
{
    index->data()->axis(0)->set_tiling(std::vector<Index>{1, 2});
}

//! Vocab (8, 8).
inline void nn_pytorch_tile_vocab_8x8(graph::NNGraph::TensorNode* vocab)
{
    vocab->data()->axis(0)->set_tiling(std::vector<Index>{3, 2, 3});
    vocab->data()->axis(1)->set_tiling(std::vector<Index>{3, 5});
}

//! Softmax along axis 0 on (6, 7): heterogeneous on axis 1 only; axis 0 unsplit.
inline void nn_pytorch_tile_softmax_axis0_6x7(graph::NNGraph::TensorNode* x)
{
    x->data()->axis(0)->set_tiling(std::vector<Index>{6});
    x->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
}

//! Softmax along axis 1 on (6, 7): heterogeneous on axis 0 only; axis 1 unsplit.
inline void nn_pytorch_tile_softmax_axis1_6x7(graph::NNGraph::TensorNode* x)
{
    x->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    x->data()->axis(1)->set_tiling(std::vector<Index>{7});
}


//! Rank-3 tensor shape (2, 3, 4): heterogeneous splits on every axis.
inline void nn_pytorch_tile_heterogeneous_rank3_2x3x4(graph::NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{1, 1});
    t->data()->axis(1)->set_tiling(std::vector<Index>{1, 2});
    t->data()->axis(2)->set_tiling(std::vector<Index>{2, 2});
}

//! Rank-4 `(h, s, b0, b1)` operands (e.g. SDPA Q/K/V): non-uniform splits per axis.
inline void nn_pytorch_tile_heterogeneous_rank4_hs_bn_b0b1(graph::NNGraph::TensorNode* t)
{
    for(Index d = 0; d < t->ndim(); ++d)
    {
        const Index L = t->shape()[static_cast<size_t>(d)];
        if(L >= 4)
        {
            t->data()->axis(d)->set_tiling(std::vector<Index>{1, L - 1});
        }
        else if(L == 3)
        {
            t->data()->axis(d)->set_tiling(std::vector<Index>{1, 2});
        }
        else if(L == 2)
        {
            t->data()->axis(d)->set_tiling(std::vector<Index>{1, 1});
        }
        else
        {
            t->data()->axis(d)->set_tiling(std::vector<Index>{L});
        }
    }
}

//! Boolean mask `(n, n)` with non-uniform row/col tiling when possible.
inline void nn_pytorch_tile_mask_nn(graph::NNGraph::TensorNode* mask)
{
    for(Index d = 0; d < mask->ndim(); ++d)
    {
        const Index L = mask->shape()[static_cast<size_t>(d)];
        if(L >= 4)
        {
            mask->data()->axis(d)->set_tiling(std::vector<Index>{1, L - 1});
        }
        else if(L == 3)
        {
            mask->data()->axis(d)->set_tiling(std::vector<Index>{1, 2});
        }
        else if(L == 2)
        {
            mask->data()->axis(d)->set_tiling(std::vector<Index>{1, 1});
        }
        else
        {
            mask->data()->axis(d)->set_tiling(std::vector<Index>{L});
        }
    }
}

//! RoPE: `sin`, `cos`, `src` layouts compatible with `src.shape[0] == 2*sin.shape[0]`.
inline void nn_pytorch_tile_rope_sin_cos_src(
    graph::NNGraph::TensorNode* sin,
    graph::NNGraph::TensorNode* cos,
    graph::NNGraph::TensorNode* src)
{
    for(Index d = 0; d < sin->ndim(); ++d)
    {
        const Index Ls = sin->shape()[static_cast<size_t>(d)];
        std::vector<Index> sin_seg;
        if(Ls >= 4)
        {
            sin_seg = {1, Ls - 1};
        }
        else if(Ls == 3)
        {
            sin_seg = {1, 2};
        }
        else if(Ls == 2)
        {
            sin_seg = {1, 1};
        }
        else
        {
            sin_seg = {Ls};
        }
        sin->data()->axis(d)->set_tiling(sin_seg);
        cos->data()->axis(d)->set_tiling(sin_seg);
        if(d == 0)
        {
            std::vector<Index> src_seg;
            src_seg.reserve(sin_seg.size());
            for(Index v : sin_seg)
            {
                src_seg.push_back(2 * v);
            }
            src->data()->axis(0)->set_tiling(std::move(src_seg));
        }
        else
        {
            src->data()->axis(d)->set_tiling(sin_seg);
        }
    }
}

} // namespace nntile::test

#endif // NNTILE_HAVE_TORCH
