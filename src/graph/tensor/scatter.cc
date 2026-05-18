/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scatter.cc
 * TensorGraph scatter operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scatter.hh"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/copy.hh"
#include "nntile/graph/tile/copy_intersection.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/scatter.hh"

namespace nntile::graph::tensor
{

void TensorScatterOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::scatter_async (src/tensor/scatter.cc).
    const TensorAxisLayout* lay_dst = ctx.tiling.find(dst);
    if(lay_dst == nullptr)
    {
        throw std::runtime_error("lower_to_tile SCATTER: missing tiling for dst");
    }
    const auto& tsrc = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tdst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tsrc.size() != 1)
    {
        throw std::runtime_error(
            "lower_to_tile SCATTER: src must be single-tile tensor");
    }
    TileGraph::TileNode* src_tile = tsrc[0];
    const Index ndim = dst->ndim();
    if(tdst.size() == 1)
    {
        tile_graph::copy(src_tile, tdst[0]);
        return;
    }
    const std::string scratch_name = std::string("__scatter_scr_") +
        std::to_string(reinterpret_cast<std::uintptr_t>(this));
    TileGraph::TileNode* scratch = ctx.out.data(
        std::vector<Index>{2 * ndim}, scratch_name, DataType::INT64);
    std::vector<Index> src_corner(static_cast<size_t>(ndim), 0);
    std::vector<Index> dst_corner(static_cast<size_t>(ndim));
    std::vector<Index> grid_coord;
    for(Index lin = 0; lin < lay_dst->grid_volume(); ++lin)
    {
        lay_dst->grid_coord_from_linear(lin, grid_coord);
        for(Index k = 0; k < ndim; ++k)
        {
            Index lo = 0;
            Index hi = 0;
            lay_dst->tile_axis_global_range(grid_coord, k, lo, hi);
            dst_corner[static_cast<size_t>(k)] = lo;
        }
        tile_graph::copy_intersection(src_tile, src_corner,
            tdst[static_cast<size_t>(lin)], dst_corner, scratch);
    }
}

void scatter(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "scatter: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "scatter: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "scatter: input tensors must have the same dtype");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "scatter: src and dst must have the same shape");
    }

    auto op = std::make_shared<TensorScatterOp>(src, dst);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
