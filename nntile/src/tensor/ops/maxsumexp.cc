#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/maxsumexp.cc
 * TensorGraph maxsumexp operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/maxsumexp.hh"

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/ops/clear.hh"
#include "nntile/tile/ops/maxsumexp.hh"
#include "nntile/tensor/ops/clear.hh"
#include "nntile/tensor/ops/maxsumexp.hh"

#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

namespace
{

//! Map maxsumexp grid coord to src grid coord (omit pair axis).
void maxsumexp_to_src_grid_coord(const std::vector<Index> &m_coord,
    Index axis,
    Index src_ndim,
    std::vector<Index> &src_coord)
{
    src_coord.resize(static_cast<size_t>(src_ndim));
    for (Index g = 0; g < src_ndim; ++g)
    {
        if (g == axis)
        {
            continue;
        }
        const Index m_g = (g < axis) ? g : (g - 1);
        const Index m_ndim = static_cast<Index>(m_coord.size());
        src_coord[static_cast<size_t>(layout_axis(g, src_ndim))] =
            m_coord[static_cast<size_t>(layout_axis(m_g, m_ndim))];
    }
}

} // namespace

TensorGraph::TensorNode *maxsumexp(
    TensorGraph::TensorNode *src, Index axis, int redux)
{
    if (src == nullptr)
    {
        throw std::invalid_argument(
            "maxsumexp: input tensor must be non-null");
    }

    // C-order: same ndim as src, last dim = 2, axis removed from interior.
    std::vector<Index> output_shape;
    output_shape.reserve(src->ndim());
    for (Index i = 0; i < src->ndim(); ++i)
    {
        if (i != axis)
        {
            output_shape.push_back(src->shape()[i]);
        }
    }
    output_shape.push_back(2);

    TensorGraph::TensorNode *dst =
        src->graph()->data(std::move(output_shape), src->dtype());

    validate_maxsumexp_shape_and_merge(src, dst, axis, "maxsumexp");

    auto op = std::make_shared<TensorMaxsumexpOp>(src, dst, axis, redux);
    src->graph()->add_op(op);

    return dst;
}

void maxsumexp(TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    Index axis,
    int redux)
{
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "maxsumexp: input tensors must be non-null");
    }
    if (src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "maxsumexp: input tensors must belong to the same graph");
    }
    if (src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "maxsumexp: input tensors must have the same dtype");
    }
    validate_maxsumexp_shape_and_merge(src, dst, axis, "maxsumexp");

    auto op = std::make_shared<TensorMaxsumexpOp>(src, dst, axis, redux);
    src->graph()->add_op(op);
}

void TensorMaxsumexpOp::lower_to_tile(const LoweringContext &ctx) const
{
    const TensorAxisLayout *lay_src = ctx.tiling.find(src);
    const TensorAxisLayout *lay_dst = ctx.tiling.find(dst);
    if (lay_src == nullptr || lay_dst == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile MAXSUMEXP: missing tiling for src and/or dst");
    }

    const auto &tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto &tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index src_nd = src->ndim();
    const Index lay_ax = layout_axis(axis, src_nd);

    std::vector<Index> dst_coord;
    std::vector<Index> src_coord(static_cast<size_t>(src_nd));

    for (Index lin_dst = 0; lin_dst < lay_dst->grid_volume(); ++lin_dst)
    {
        lay_dst->grid_coord_from_linear(lin_dst, dst_coord);
        TileGraph::TileNode *dst_tile =
            tiles_dst[static_cast<size_t>(lin_dst)];

        maxsumexp_to_src_grid_coord(dst_coord, axis, src_nd, src_coord);

        tile::clear(dst_tile);

        const Index nseg_along_axis =
            lay_src->grid_shape()[static_cast<size_t>(lay_ax)];
        for (Index j = 0; j < nseg_along_axis; ++j)
        {
            src_coord[static_cast<size_t>(lay_ax)] = j;
            const Index lin_src = lay_src->grid_linear(src_coord);
            TileGraph::TileNode *src_tile =
                tiles_src[static_cast<size_t>(lin_src)];
            tile::maxsumexp(src_tile, dst_tile, axis, redux);
        }
    }
}

} // namespace nntile::tensor
