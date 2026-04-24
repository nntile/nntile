/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/embedding.cc
 * TensorGraph embedding operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/embedding.hh"

#include <stdexcept>
#include <vector>

#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/clear.hh"
#include "nntile/graph/tile/embedding.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/embedding.hh"
#include "nntile/tile/traits.hh"

namespace nntile::graph::tensor
{

namespace
{

//! All tiles along `dim` must have the same extent (matches tensor basetile assumptions).
Index uniform_tile_extent_along(
    const nntile::graph::TensorAxisLayout& lay, Index dim, const char* op)
{
    const auto& gs = lay.grid_shape();
    if(dim < 0 || dim >= static_cast<Index>(gs.size()))
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": uniform_tile_extent_along: bad dim");
    }
    Index first = -1;
    std::vector<Index> coord(gs.size(), 0);
    for(Index lin = 0; lin < lay.grid_volume(); ++lin)
    {
        lay.grid_coord_from_linear(lin, coord);
        const Index ext = lay.tile_shape_at(coord)[static_cast<size_t>(dim)];
        if(first < 0)
        {
            first = ext;
        }
        else if(ext != first)
        {
            throw std::runtime_error(std::string("lower_to_tile ") + op +
                ": non-uniform tile extent along embedding/vocab axis");
        }
    }
    return first;
}

} // namespace

TensorGraph::TensorNode* embedding(TensorGraph::TensorNode* index,
                                    TensorGraph::TensorNode* vocab,
                                    const std::string& output_name,
                                    Index axis)
{
    if(index == nullptr || vocab == nullptr)
        throw std::invalid_argument("embedding: tensors must be non-null");
    if(index->graph() != vocab->graph())
        throw std::invalid_argument("embedding: tensors must belong to same graph");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding: index must have INT64 dtype");
    // Output shape: index.shape + (vocab.shape[0],) at axis
    // NNTile layout: vocab [embed_dim, num_embeddings]; embed.shape[axis] == vocab.shape[0]
    std::vector<Index> embed_shape = index->shape();
    if(vocab->ndim() != 2)
        throw std::invalid_argument("embedding: vocab must be 2D");
    embed_shape.push_back(vocab->dim(0));
    TensorGraph::TensorNode* embed = vocab->graph()->data(
        std::move(embed_shape), output_name, vocab->dtype());

    embedding(index, vocab, embed, axis);
    return embed;
}

void embedding(TensorGraph::TensorNode* index,
               TensorGraph::TensorNode* vocab,
               TensorGraph::TensorNode* embed,
               Index axis)
{
    if(index == nullptr || vocab == nullptr || embed == nullptr)
        throw std::invalid_argument("embedding: tensors must be non-null");
    if(index->graph() != vocab->graph() || vocab->graph() != embed->graph())
        throw std::invalid_argument("embedding: tensors must belong to same graph");
    if(index->dtype() != DataType::INT64)
        throw std::invalid_argument("embedding: index must have INT64 dtype");
    if(vocab->dtype() != embed->dtype())
        throw std::invalid_argument("embedding: vocab and embed must have same dtype");
    validate_embedding_shape_and_merge(embed, index, vocab, "embedding");

    auto op = std::make_shared<TensorEmbeddingOp>(index, vocab, embed, axis);
    embed->graph()->add_op(op);
}

void TensorEmbeddingOp::lower_to_tile(const LoweringContext& ctx) const
{
    const TensorAxisLayout* lay_e = ctx.tiling.find(embed);
    const TensorAxisLayout* lay_i = ctx.tiling.find(index);
    const TensorAxisLayout* lay_v = ctx.tiling.find(vocab);
    if(lay_e == nullptr || lay_i == nullptr || lay_v == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile EMBEDDING: missing tiling for index/vocab/embed");
    }

    const Index vocab_b0 =
        uniform_tile_extent_along(*lay_v, 0, "EMBEDDING");
    const Index embed_axis_bs =
        uniform_tile_extent_along(*lay_e, axis, "EMBEDDING");
    if(embed_axis_bs % vocab_b0 != 0)
    {
        throw std::runtime_error(
            "lower_to_tile EMBEDDING: embed tile extent along axis must be "
            "divisible by vocab tile extent along dim 0");
    }

    const auto& tiles_i = tile_lower::tiles_of(ctx.tile_map, index);
    const auto& tiles_v = tile_lower::tiles_of(ctx.tile_map, vocab);
    const auto& tiles_e = tile_lower::tiles_of(ctx.tile_map, embed);

    if(static_cast<Index>(tiles_e.size()) != lay_e->grid_volume())
    {
        throw std::runtime_error(
            "lower_to_tile EMBEDDING: embed tile count mismatch");
    }

    std::vector<Index> embed_coord;
    std::vector<Index> index_coord;
    const Index g1_vocab =
        lay_v->grid_shape().size() > 1 ? lay_v->grid_shape()[1] : 1;

    for(Index lin_e = 0; lin_e < lay_e->grid_volume(); ++lin_e)
    {
        lay_e->grid_coord_from_linear(lin_e, embed_coord);
        tile_graph::clear(tiles_e[static_cast<size_t>(lin_e)]);

        index_coord.resize(static_cast<size_t>(index->ndim()));
        for(Index j = 0; j < axis; ++j)
        {
            index_coord[static_cast<size_t>(j)] = embed_coord[static_cast<size_t>(j)];
        }
        for(Index j = axis; j < index->ndim(); ++j)
        {
            index_coord[static_cast<size_t>(j)] =
                embed_coord[static_cast<size_t>(j + 1)];
        }
        const Index lin_i = lay_i->grid_linear(index_coord);
        TileGraph::TileNode* index_tile =
            tiles_i[static_cast<size_t>(lin_i)];

        Index axis_lo = 0, axis_hi = 0;
        lay_e->tile_axis_global_range(embed_coord, axis, axis_lo, axis_hi);
        const Index vocab_tile0_start = axis_lo / vocab_b0;

        const auto embed_ts = lay_e->tile_shape_at(embed_coord);
        const Index k_axis = embed_ts[static_cast<size_t>(axis)];
        const Index vocab_span =
            (k_axis - 1) / vocab_b0 + 1;

        nntile::tile::TileTraits embed_traits(embed_ts);
        const Index m = embed_traits.stride[axis];
        const Index n = embed_traits.matrix_shape[static_cast<size_t>(axis) + 1][1];
        const Index k = embed_traits.shape[axis];

        const Index vocab_g0 = lay_v->grid_shape()[0];
        for(Index tv0 = vocab_tile0_start;
            tv0 < vocab_tile0_start + vocab_span && tv0 < vocab_g0;
            ++tv0)
        {
            for(Index tv1 = 0; tv1 < g1_vocab; ++tv1)
            {
                std::vector<Index> vocab_coord = {tv0, tv1};
                const Index lin_v = lay_v->grid_linear(vocab_coord);
                TileGraph::TileNode* vocab_tile =
                    tiles_v[static_cast<size_t>(lin_v)];
                const auto vocab_ts = lay_v->tile_shape_at(vocab_coord);
                nntile::tile::TileTraits vocab_traits(vocab_ts);

                const Index k_start = (tv0 - vocab_tile0_start) * vocab_b0;
                const Index k_size = vocab_traits.shape[0];
                tile_graph::embedding(
                    m,
                    n,
                    k,
                    k_start,
                    k_size,
                    index_tile,
                    vocab_tile,
                    tiles_e[static_cast<size_t>(lin_e)]);
            }
        }
    }
}

} // namespace nntile::graph::tensor
