#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/norm_slice_inplace.cc
 * TensorGraph norm_slice_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/norm_slice_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/norm_slice_inplace.hh"
#include "nntile/tensor/ops/norm_slice_inplace.hh"

namespace nntile::tensor
{

void TensorNormSliceInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::norm_slice_inplace_async (src/tensor/norm_slice_inplace.cc).
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    if(lay_d == nullptr || lay_s == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile NORM_SLICE_INPLACE: missing tiling for dst and/or "
            "src");
    }
    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);
    constexpr Scalar one = 1.0;
    std::vector<Index> dst_coord;
    std::vector<Index> s_coord(static_cast<size_t>(src->ndim()));
    const Index src_nd = src->ndim();
    const Index dst_nd = dst->ndim();
    const Index s_axis = graph_axis_to_storage(axis, src_nd);
    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        for(Index sd = 0; sd < dst_nd; ++sd)
        {
            const Index g_dst = storage_axis_to_graph(sd, dst_nd);
            Index g_src = 0;
            Index k = 0;
            for(Index g2 = 0; g2 < src_nd; ++g2)
            {
                if(g2 == axis)
                {
                    continue;
                }
                if(k == g_dst)
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
        for(Index jj = 0; jj < nseg_along_axis; ++jj)
        {
            s_coord[static_cast<size_t>(s_axis)] = jj;
            const Index lin_s = lay_s->grid_linear(s_coord);
            if(jj == 0)
            {
                tile::norm_slice_inplace(
                    alpha,
                    tiles_s[static_cast<size_t>(lin_s)],
                    beta,
                    tiles_d[static_cast<size_t>(lin_d)],
                    axis,
                    redux);
            }
            else
            {
                tile::norm_slice_inplace(
                    alpha,
                    tiles_s[static_cast<size_t>(lin_s)],
                    one,
                    tiles_d[static_cast<size_t>(lin_d)],
                    axis,
                    redux);
            }
        }
    }
}

void norm_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_slice_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "norm_slice_inplace: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(dst, src, axis, "norm_slice_inplace");

    auto op = std::make_shared<TensorNormSliceInplaceOp>(
        alpha, beta, src, dst, axis, redux);
    src->graph()->add_op(op);
}

} // namespace nntile::tensor
