/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gather.cc
 * TensorGraph gather operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gather.hh"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/copy.hh"
#include "nntile/graph/tile/copy_intersection.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/gather.hh"

namespace nntile::graph::tensor
{

void TensorGatherOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::gather_async (src/tensor/gather.cc).
    const TensorAxisLayout* lay_src = ctx.tiling.find(src);
    if(lay_src == nullptr)
    {
        throw std::runtime_error("lower_to_tile GATHER: missing tiling for src");
    }
    const auto& tsrc = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tdst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tdst.size() != 1)
    {
        throw std::runtime_error(
            "lower_to_tile GATHER: dst must be single-tile tensor");
    }
    TileGraph::TileNode* dst_tile = tdst[0];
    const Index ndim = src->ndim();
    if(tsrc.size() == 1)
    {
        tile_graph::copy(tsrc[0], dst_tile);
        return;
    }
    const std::string scratch_name = std::string("__gather_scr_") +
        std::to_string(reinterpret_cast<std::uintptr_t>(this));
    TileGraph::TileNode* scratch = ctx.out.data(
        std::vector<Index>{2 * ndim}, scratch_name, DataType::INT64);
    std::vector<Index> src_corner(static_cast<size_t>(ndim));
    std::vector<Index> dst_corner(static_cast<size_t>(ndim), 0);
    std::vector<Index> grid_coord;
    for(Index lin = 0; lin < lay_src->grid_volume(); ++lin)
    {
        lay_src->grid_coord_from_linear(lin, grid_coord);
        for(Index k = 0; k < ndim; ++k)
        {
            Index lo = 0;
            Index hi = 0;
            lay_src->tile_axis_global_range(grid_coord, k, lo, hi);
            src_corner[static_cast<size_t>(k)] = lo;
        }
        tile_graph::copy_intersection(tsrc[static_cast<size_t>(lin)], src_corner,
            dst_tile, dst_corner, scratch);
    }
}

TensorGraph::TensorNode* gather(
    TensorGraph::TensorNode* src,
    const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "gather: input tensor must be non-null");
    }

    std::vector<Index> output_shape = src->shape();
    TensorGraph::TensorNode* dst = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());

    gather(src, dst);

    return dst;
}

void gather(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "gather: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "gather: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "gather: input tensors must have the same dtype");
    }
    if(src->shape() != dst->shape())
    {
        throw std::invalid_argument(
            "gather: src and dst must have the same shape");
    }

    auto op = std::make_shared<TensorGatherOp>(src, dst);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
