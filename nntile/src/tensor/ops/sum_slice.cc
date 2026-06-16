#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/sum_slice.cc
 * TensorGraph sum_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/sum_slice.hh"

#include "nntile/base_types.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/sum_slice.hh"
#include "nntile/tensor/ops/sum_slice.hh"

#include <stdexcept>
#include <utility>
#include <vector>

namespace nntile::tensor
{

namespace
{

std::vector<Index> sum_slice_output_shape(
    const std::vector<Index> &src_shape, Index axis)
{
    std::vector<Index> out_shape;
    out_shape.reserve(src_shape.size() - 1);
    for (Index i = 0; i < src_shape.size(); ++i)
    {
        if (i != axis)
        {
            out_shape.push_back(src_shape[i]);
        }
    }
    return out_shape;
}

} // namespace

TensorGraph::TensorNode *sum_slice(TensorGraph::TensorNode *src,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if (src == nullptr)
    {
        throw std::invalid_argument(
            "sum_slice: input tensor must be non-null");
    }
    if (axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument("sum_slice: axis out of range");
    }

    std::vector<Index> output_shape =
        sum_slice_output_shape(src->shape(), axis);
    TensorGraph::TensorNode *output =
        src->graph()->data(std::move(output_shape), src->dtype());

    validate_slice_shape_and_merge(output, src, axis, "sum_slice");

    auto op = std::make_shared<TensorSumSliceOp>(
        src, output, axis, redux, alpha, beta);
    src->graph()->add_op(op);

    return output;
}

void sum_slice(TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if (src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must be non-null");
    }
    if (src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must belong to the same graph");
    }
    if (src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sum_slice: input tensors must have the same dtype");
    }
    if (axis < 0 || axis >= src->ndim())
    {
        throw std::invalid_argument("sum_slice: axis out of range");
    }
    if (src == dst)
    {
        throw std::invalid_argument(
            "sum_slice: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(dst, src, axis, "sum_slice");

    auto op =
        std::make_shared<TensorSumSliceOp>(src, dst, axis, redux, alpha, beta);
    src->graph()->add_op(op);
}

void TensorSumSliceOp::lower_to_tile(const LoweringContext &ctx) const
{
    // Match nntile::tensor::sum_slice_async (src/tensor/sum_slice.cc).
    const TensorAxisLayout *lay_s = ctx.tiling.find(src);
    const TensorAxisLayout *lay_d = ctx.tiling.find(dst);
    if (lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SUM_SLICE: missing tiling for src and/or dst");
    }

    const auto &tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto &tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index src_nd = src->ndim();
    const Index dst_nd = dst->ndim();
    const Index s_axis = graph_axis_to_storage(axis, src_nd);

    std::vector<Index> dst_coord;
    std::vector<Index> s_coord(static_cast<size_t>(src_nd));

    for (Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        TileGraph::TileNode *dst_tile = tiles_d[static_cast<size_t>(lin_d)];

        for (Index sd = 0; sd < dst_nd; ++sd)
        {
            const Index g_dst = storage_axis_to_graph(sd, dst_nd);
            Index g_src = 0;
            Index k = 0;
            for (Index g2 = 0; g2 < src_nd; ++g2)
            {
                if (g2 == axis)
                {
                    continue;
                }
                if (k == g_dst)
                {
                    g_src = g2;
                    break;
                }
                ++k;
            }
            s_coord[static_cast<size_t>(
                graph_axis_to_storage(g_src, src_nd))] =
                dst_coord[static_cast<size_t>(sd)];
        }

        const Index nseg_along_axis =
            lay_s->grid_shape()[static_cast<size_t>(s_axis)];

        s_coord[static_cast<size_t>(s_axis)] = 0;
        Index lin_s0 = lay_s->grid_linear(s_coord);
        tile::sum_slice(alpha,
            tiles_s[static_cast<size_t>(lin_s0)],
            beta,
            dst_tile,
            axis,
            redux);

        for (Index jj = 1; jj < nseg_along_axis; ++jj)
        {
            s_coord[static_cast<size_t>(s_axis)] = jj;
            const Index lin_s = lay_s->grid_linear(s_coord);
            tile::sum_slice(alpha,
                tiles_s[static_cast<size_t>(lin_s)],
                Scalar(1.0),
                dst_tile,
                axis,
                redux);
        }
    }
}

} // namespace nntile::tensor
