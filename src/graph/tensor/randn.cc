/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/randn.cc
 * TensorGraph randn operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/randn.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/randn.hh"
#include "nntile/tensor/randn.hh"

namespace nntile::graph::tensor
{

void TensorRandnOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::randn_async tile loop (src/tensor/randn.cc).
    const TensorAxisLayout* lay = ctx.tiling.find(dst);
    if(lay == nullptr)
    {
        throw std::runtime_error("lower_to_tile RANDN: missing tiling for dst");
    }
    const auto& tiles = tile_lower::tiles_of(ctx.tile_map, dst);
    if(static_cast<size_t>(lay->grid_volume()) != tiles.size())
    {
        throw std::runtime_error("lower_to_tile RANDN: tile count mismatch");
    }
    std::vector<Index> grid_coord;
    std::vector<Index> tile_start(static_cast<size_t>(dst->ndim()));
    for(Index lin = 0; lin < lay->grid_volume(); ++lin)
    {
        lay->grid_coord_from_linear(lin, grid_coord);
        for(Index d = 0; d < dst->ndim(); ++d)
        {
            Index glo = 0;
            Index ghi = 0;
            lay->tile_axis_global_range(grid_coord, d, glo, ghi);
            tile_start[static_cast<size_t>(d)] = glo;
        }
        tile_graph::randn(tiles[static_cast<size_t>(lin)], tile_start,
            underlying_shape, seed, mean, stddev);
    }
}

void randn(
    TensorGraph::TensorNode* dst,
    const std::vector<Index>& start,
    const std::vector<Index>& underlying_shape,
    unsigned long long seed,
    Scalar mean,
    Scalar stddev)
{
    if(dst == nullptr)
    {
        throw std::invalid_argument(
            "randn: dst tensor must be non-null");
    }
    if(start.size() != underlying_shape.size())
    {
        throw std::invalid_argument(
            "randn: start and underlying_shape must have same size");
    }
    if(dst->ndim() != static_cast<Index>(start.size()))
    {
        throw std::invalid_argument(
            "randn: start size must match dst ndim");
    }

    auto op = std::make_shared<TensorRandnOp>(
        dst, start, underlying_shape, seed, mean, stddev);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
