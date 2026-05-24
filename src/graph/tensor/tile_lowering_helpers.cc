/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/tile_lowering_helpers.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/tile_lowering_helpers.hh"

#include <string>

#include "nntile/graph/tile/graph_ops.hh"

namespace nntile::graph::tile_lower
{

void assert_same_elementwise_layout(
    const TensorGraph::TensorNode* a,
    const TensorGraph::TensorNode* b,
    const char* ctx)
{
    if(a->shape() != b->shape())
    {
        throw std::invalid_argument(std::string(ctx) + ": shape mismatch");
    }
    if(a->ndim() != b->ndim())
    {
        throw std::invalid_argument(std::string(ctx) + ": ndim mismatch");
    }
    for(Index d = 0; d < a->ndim(); ++d)
    {
        if(a->axis(static_cast<int>(d))->tile_sizes !=
           b->axis(static_cast<int>(d))->tile_sizes)
        {
            throw std::invalid_argument(
                std::string(ctx) + ": tiling mismatch on axis " +
                std::to_string(d));
        }
    }
}

const std::vector<TileGraph::TileNode*>& tiles_of(
    const TensorNodeToTileMap& m,
    const TensorGraph::TensorNode* n)
{
    auto it = m.find(n);
    if(it == m.end())
    {
        throw std::runtime_error(
            "lower_to_tile: missing tile map for tensor '" + n->name() + "'");
    }
    return it->second;
}

std::vector<TileGraph::TileNode*> copy_tiles(
    const TensorNodeToTileMap& m,
    const TensorGraph::TensorNode* n)
{
    const auto& v = tiles_of(m, n);
    return std::vector<TileGraph::TileNode*>(v.begin(), v.end());
}

void lower_unary2(
    const TensorGraph::TensorNode* src,
    const TensorGraph::TensorNode* dst,
    const TensorNodeToTileMap& m,
    const char* ctx,
    void (*fn)(TileGraph::TileNode*, TileGraph::TileNode*))
{
    const auto& vs = tiles_of(m, src);
    const auto& vd = tiles_of(m, dst);
    if(vs.size() != vd.size())
    {
        throw std::runtime_error(
            std::string(ctx) + ": tile count mismatch");
    }
    assert_same_elementwise_layout(src, dst, ctx);
    for(size_t i = 0; i < vs.size(); ++i)
    {
        fn(vs[i], vd[i]);
    }
}

void lower_inplace1(
    const TensorGraph::TensorNode* x,
    const TensorNodeToTileMap& m,
    const char* ctx,
    void (*fn)(TileGraph::TileNode*))
{
    assert_same_elementwise_layout(x, x, ctx);
    for(TileGraph::TileNode* t : tiles_of(m, x))
    {
        fn(t);
    }
}

void lower_backward3(
    const TensorGraph::TensorNode* x,
    const TensorGraph::TensorNode* dy,
    const TensorGraph::TensorNode* dx,
    const TensorNodeToTileMap& m,
    const char* ctx,
    void (*fn)(TileGraph::TileNode*, TileGraph::TileNode*, TileGraph::TileNode*))
{
    const auto& vx = tiles_of(m, x);
    const auto& vdy = tiles_of(m, dy);
    const auto& vdx = tiles_of(m, dx);
    if(vx.size() != vdy.size() || vx.size() != vdx.size())
    {
        throw std::runtime_error(
            std::string(ctx) + ": tile count mismatch");
    }
    assert_same_elementwise_layout(x, dy, ctx);
    assert_same_elementwise_layout(x, dx, ctx);
    for(size_t i = 0; i < vx.size(); ++i)
    {
        fn(vx[i], vdy[i], vdx[i]);
    }
}

} // namespace nntile::graph::tile_lower
