#include <nntile/common.hh>
/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/tensor/ops/swap_two_axes.cc
 * TensorGraph swap_two_axes operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/swap_two_axes.hh"

#include "nntile/core/swap_two_axes_decompose.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/ops/swap_two_axes.hh"
#include "nntile/tensor/ops/swap_two_axes.hh"

#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

namespace
{

Index normalize_dim(Index dim, Index ndim)
{
    if (dim < 0)
    {
        dim += ndim;
    }
    return dim;
}

void validate_swap_axes(Index dim0, Index dim1, Index ndim)
{
    if (ndim < 2)
    {
        throw std::invalid_argument("swap_two_axes: rank must be >= 2");
    }
    dim0 = normalize_dim(dim0, ndim);
    dim1 = normalize_dim(dim1, ndim);
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim)
    {
        throw std::invalid_argument("swap_two_axes: axis out of range");
    }
    if (dim0 == dim1)
    {
        throw std::invalid_argument("swap_two_axes: axes must differ");
    }
}

void merge_axes_for_swap(
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    Index dim0,
    Index dim1)
{
    const Index n = src->ndim();
    for (Index i = 0; i < n; ++i)
    {
        if (i == dim0)
        {
            merge_axis(
                src->mutable_axes()[static_cast<size_t>(dim1)],
                dst->mutable_axes()[static_cast<size_t>(dim0)]);
        }
        else if (i == dim1)
        {
            merge_axis(
                src->mutable_axes()[static_cast<size_t>(dim0)],
                dst->mutable_axes()[static_cast<size_t>(dim1)]);
        }
        else
        {
            merge_axis(
                src->mutable_axes()[static_cast<size_t>(i)],
                dst->mutable_axes()[static_cast<size_t>(i)]);
        }
    }
}

} // namespace

void TensorSwapTwoAxesOp::lower_to_tile(const LoweringContext &ctx) const
{
    const TensorAxisLayout *lay_s = ctx.tiling.find(src);
    const TensorAxisLayout *lay_d = ctx.tiling.find(dst);
    if (lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SWAP_TWO_AXES: missing tiling for src or dst");
    }
    const auto &tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto &tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);
    const Index nd = src->ndim();
    std::vector<Index> src_coord;
    std::vector<Index> dst_coord(static_cast<size_t>(nd));
    for (Index lin_s = 0; lin_s < lay_s->grid_volume(); ++lin_s)
    {
        lay_s->grid_coord_from_linear(lin_s, src_coord);
        for (Index d = 0; d < nd; ++d)
        {
            dst_coord[static_cast<size_t>(d)] = src_coord[static_cast<size_t>(d)];
        }
        std::swap(
            dst_coord[static_cast<size_t>(dim0)],
            dst_coord[static_cast<size_t>(dim1)]);
        const Index lin_d = lay_d->grid_linear(dst_coord);
        tile::swap_two_axes(
            tiles_s[static_cast<size_t>(lin_s)],
            tiles_d[static_cast<size_t>(lin_d)],
            dim0,
            dim1);
    }
}

void swap_two_axes(
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    Index dim0,
    Index dim1)
{
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("swap_two_axes: tensors must be non-null");
    }
    if (src == dst)
    {
        throw std::invalid_argument(
            "swap_two_axes: src and dst must be distinct tensors");
    }
    if (src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "swap_two_axes: tensors must belong to same graph");
    }
    if (src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "swap_two_axes: tensors must have the same dtype");
    }
    const Index n = src->ndim();
    validate_swap_axes(dim0, dim1, n);
    dim0 = normalize_dim(dim0, n);
    dim1 = normalize_dim(dim1, n);
    if (dim0 > dim1)
    {
        std::swap(dim0, dim1);
    }

    const std::vector<Index> src_shape = src->shape();
    const core::SwapTwoAxesDecomposition decomp =
        core::decompose_swap_axes(src_shape, dim0, dim1);
    const auto &dst_shape = decomp.output_shape;
    if (dst->shape() != dst_shape)
    {
        throw std::invalid_argument("swap_two_axes: dst shape mismatch");
    }
    if (dst->ndim() != n)
    {
        throw std::invalid_argument("swap_two_axes: dst.ndim must equal src.ndim");
    }

    merge_axes_for_swap(src, dst, dim0, dim1);
    auto op = std::make_shared<TensorSwapTwoAxesOp>(src, dst, dim0, dim1);
    src->graph()->add_op(op);
}

} // namespace nntile::tensor
