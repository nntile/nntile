/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/tests/nn_graph/pytorch_tile_helpers.hh
 * Heterogeneous per-axis tiling helpers for NNGraph PyTorch + TileGraph tests.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_HAVE_TORCH

#   include <stdexcept>
#   include <vector>

#   include <nntile/common.hh>
#   include <nntile/nn/graph.hh>
#   include <nntile/tensor/graph.hh>

namespace nntile::test
{

using nntile::Index;

//! 2D tensor graph shape (6, 7).
inline void nn_pytorch_tile_heterogeneous_rank2_6x7(NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    t->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
}

//! 1D tensor length 7.
inline void nn_pytorch_tile_heterogeneous_1d_len7(NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{3, 4});
}

//! 1D tensor length 6.
inline void nn_pytorch_tile_heterogeneous_1d_len6(NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 4});
}

//! 1D tensor length 4.
inline void nn_pytorch_tile_heterogeneous_1d_len4(NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 2});
}

//! Logits (batch, nclasses) = (7, 5) graph shape.
//! Class axis (last) is a single tile for `subtract_indexed_outputs`.
inline void nn_pytorch_tile_logits_5x7(NNGraph::TensorNode* x)
{
    x->data()->axis(0)->set_tiling(std::vector<Index>{3, 4});
    x->data()->axis(1)->set_tiling(std::vector<Index>{5});
}

//! GEMM x (N,K)=(6,7), w (K,M)=(7,6) graph shapes.
inline void nn_pytorch_tile_gemm_operands_6_7_6(
    NNGraph::TensorNode* a, NNGraph::TensorNode* b)
{
    a->data()->axis(0)->set_tiling(std::vector<Index>{3, 4});
    a->data()->axis(1)->set_tiling(std::vector<Index>{2, 4});
    b->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    b->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
}

//! Vocab matrix (num_embeddings, embed_dim) = (10, 10) graph shape.
//! Axis 0 (vocab ids) must not be split; axis 1 (embed_dim) uniform tiles.
inline void nn_pytorch_tile_vocab_10x10(NNGraph::TensorNode* vocab)
{
    vocab->data()->axis(0)->set_tiling(std::vector<Index>{10});
    vocab->data()->axis(1)->set_tiling(std::vector<Index>{5, 5});
}

//! Index tensor (4, 5).
inline void nn_pytorch_tile_index_4x5(NNGraph::TensorNode* index)
{
    index->data()->axis(0)->set_tiling(std::vector<Index>{2, 2});
    index->data()->axis(1)->set_tiling(std::vector<Index>{2, 3});
}

//! Index vector length 3.
inline void nn_pytorch_tile_index_len3(NNGraph::TensorNode* index)
{
    index->data()->axis(0)->set_tiling(std::vector<Index>{1, 2});
}

//! Vocab (num_embeddings, embed_dim) = (8, 8).
inline void nn_pytorch_tile_vocab_8x8(NNGraph::TensorNode* vocab)
{
    vocab->data()->axis(0)->set_tiling(std::vector<Index>{8});
    vocab->data()->axis(1)->set_tiling(std::vector<Index>{4, 4});
}

//! Softmax along graph axis 0 on shape (6, 7).
inline void nn_pytorch_tile_softmax_axis0_6x7(NNGraph::TensorNode* x)
{
    x->data()->axis(0)->set_tiling(std::vector<Index>{6});
    x->data()->axis(1)->set_tiling(std::vector<Index>{3, 4});
}

//! Softmax along graph axis 1 on shape (6, 7).
inline void nn_pytorch_tile_softmax_axis1_6x7(NNGraph::TensorNode* x)
{
    x->data()->axis(0)->set_tiling(std::vector<Index>{2, 3, 1});
    x->data()->axis(1)->set_tiling(std::vector<Index>{7});
}

//! Rank-3 tensor shape (4, 3, 2): heterogeneous splits on every axis.
inline void nn_pytorch_tile_heterogeneous_rank3_2x3x4(NNGraph::TensorNode* t)
{
    t->data()->axis(0)->set_tiling(std::vector<Index>{2, 2});
    t->data()->axis(1)->set_tiling(std::vector<Index>{1, 2});
    t->data()->axis(2)->set_tiling(std::vector<Index>{1, 1});
}

//! Rank-4 `[batch..., seq, head]` operands (e.g. SDPA Q/K/V): non-uniform splits.
inline void nn_pytorch_tile_heterogeneous_rank4_hs_bn_b0b1(NNGraph::TensorNode* t)
{
    const Index ndim = t->ndim();
    for(Index g_axis = 0; g_axis < ndim; ++g_axis)
    {
        const Index L = t->shape()[static_cast<size_t>(g_axis)];
        if(L >= 4)
        {
            t->data()->axis(g_axis)->set_tiling(std::vector<Index>{1, L - 1});
        }
        else if(L == 3)
        {
            t->data()->axis(g_axis)->set_tiling(std::vector<Index>{1, 2});
        }
        else if(L == 2)
        {
            t->data()->axis(g_axis)->set_tiling(std::vector<Index>{1, 1});
        }
        else
        {
            t->data()->axis(g_axis)->set_tiling(std::vector<Index>{L});
        }
    }
}

//! Boolean mask `(n, n)` with non-uniform row/col tiling when possible.
inline void nn_pytorch_tile_mask_nn(NNGraph::TensorNode* mask)
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

//! RoPE: `sin`, `cos`, `src` with `src` last dim == 2*sin` last dim.
inline void nn_pytorch_tile_rope_sin_cos_src(
    NNGraph::TensorNode* sin,
    NNGraph::TensorNode* cos,
    NNGraph::TensorNode* src)
{
    for(Index g_axis = 0; g_axis < sin->ndim(); ++g_axis)
    {
        const Index Ls = sin->shape()[static_cast<size_t>(g_axis)];
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
        sin->data()->axis(g_axis)->set_tiling(sin_seg);
        cos->data()->axis(g_axis)->set_tiling(sin_seg);
        if(g_axis == sin->ndim() - 1)
        {
            std::vector<Index> src_seg;
            src_seg.reserve(sin_seg.size());
            for(Index v : sin_seg)
            {
                src_seg.push_back(2 * v);
            }
            src->data()->axis(g_axis)->set_tiling(std::move(src_seg));
        }
        else
        {
            src->data()->axis(g_axis)->set_tiling(sin_seg);
        }
    }
}

//! Segment sizes for AxisDescriptor::set_tiling: positive, sum to `extent`.
inline std::vector<Index> module_heterogeneous_tile_sizes(Index extent)
{
    if(extent < 1)
    {
        throw std::invalid_argument("module_heterogeneous_tile_sizes: extent >= 1");
    }
    if(extent == 1)
    {
        return {1};
    }
    if(extent == 2)
    {
        return {1, 1};
    }
    if(extent == 3)
    {
        return {1, 2};
    }
    if(extent == 4)
    {
        return {1, 1, 2};
    }
    return {1, 2, static_cast<Index>(extent - 3)};
}

//! Heterogeneous split on every axis group that is still untiled (module / PyTorch tests).
inline void module_tile_all_untiled_axis_groups_heterogeneous(TensorGraph& tg)
{
    for(AxisDescriptor* ag : tg.axis_groups())
    {
        if(!ag->is_tiled())
        {
            ag->set_tiling(module_heterogeneous_tile_sizes(ag->extent));
        }
    }
}

//! Embedding weight layout [num_embeddings, embed_dim] graph shape.
inline void module_apply_embedding_vocab_tiling(NNGraph::TensorNode* vocab)
{
    const Index ne = vocab->shape()[0];
    const Index ed = vocab->shape()[1];
    vocab->data()->axis(0)->set_tiling(std::vector<Index>{ne});
    if(ed >= 2 && (ed % 2) == 0)
    {
        vocab->data()->axis(1)->set_tiling(std::vector<Index>{ed / 2, ed / 2});
    }
    else
    {
        vocab->data()->axis(1)->set_tiling(std::vector<Index>{ed});
    }
}

} // namespace nntile::test

#endif // NNTILE_HAVE_TORCH
