#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/tensor/ops/contiguous_view.cc
 * TensorGraph contiguous_view operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/contiguous_view.hh"

#include "nntile/tensor.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/ops/copy.hh"
#include "nntile/tensor/ops/contiguous_view.hh"

#include <stdexcept>

namespace nntile::tensor
{

namespace
{

void validate_contiguous_view(
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    const std::string &op_name)
{
    if (src->nelems() != dst->nelems())
    {
        throw std::invalid_argument(
            op_name + ": src and dst must have the same numel (" +
            std::to_string(src->nelems()) + " vs " +
            std::to_string(dst->nelems()) + ")");
    }
}

} // namespace

void contiguous_view(
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst)
{
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "contiguous_view: tensors must be non-null");
    }
    if (src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "contiguous_view: tensors must belong to same graph");
    }
    if (src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "contiguous_view: tensors must have the same dtype");
    }
    if (src == dst)
    {
        throw std::invalid_argument(
            "contiguous_view: src and dst must be distinct tensors");
    }
    validate_contiguous_view(src, dst, "contiguous_view");

    auto op = std::make_shared<TensorContiguousViewOp>(src, dst);
    src->graph()->add_op(op);
}

void TensorContiguousViewOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &src_tiles = tile_lower::tiles_of(ctx.tile_map, src);
    const auto &dst_tiles = tile_lower::tiles_of(ctx.tile_map, dst);
    if (src_tiles.size() != dst_tiles.size())
    {
        throw std::runtime_error(
            "lower_to_tile CONTIGUOUS_VIEW: tile count mismatch");
    }
    for (size_t i = 0; i < src_tiles.size(); ++i)
    {
        tile::copy_same_numel(src_tiles[i], dst_tiles[i]);
    }
}

} // namespace nntile::tensor
