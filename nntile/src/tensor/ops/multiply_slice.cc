#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/multiply_slice.cc
 * TensorGraph multiply_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/multiply_slice.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/multiply_slice.hh"

#include "nntile/tile/lowering_context.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/ops/multiply_slice.hh"

namespace nntile::tensor
{



void multiply_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_slice: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "multiply_slice: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_slice: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_slice: input tensors must have the same dtype");
    }
    validate_slice_shape_and_merge(src, dst, axis, "multiply_slice");

    auto op = std::make_shared<TensorMultiplySliceOp>(alpha, src, dst, axis);
    src->graph()->add_op(op);
}

void TensorMultiplySliceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::multiply_slice_async (src/tensor/multiply_slice.cc).
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile MULTIPLY_SLICE: missing tiling for src and/or dst");
    }

    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index nd = dst->ndim();
    const Index s_nd = src->ndim();
    const Index lay_ax = layout_axis(axis, nd);

    std::vector<Index> s_coord;
    std::vector<Index> d_coord(static_cast<size_t>(nd));

    for(Index lin_s = 0; lin_s < lay_s->grid_volume(); ++lin_s)
    {
        lay_s->grid_coord_from_linear(lin_s, s_coord);
        TileGraph::TileNode* s_tile = tiles_s[static_cast<size_t>(lin_s)];

        for(Index g = 0; g < nd; ++g)
        {
            if(g == axis)
            {
                continue;
            }
            Index g1 = 0;
            for(Index g2 = 0; g2 < nd; ++g2)
            {
                if(g2 == axis)
                {
                    continue;
                }
                if(g2 == g)
                {
                    break;
                }
                ++g1;
            }
            d_coord[static_cast<size_t>(layout_axis(g, nd))] =
                s_coord[static_cast<size_t>(layout_axis(g1, s_nd))];
        }

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(lay_ax)];
        for(Index jj = 0; jj < nseg_along_axis; ++jj)
        {
            d_coord[static_cast<size_t>(lay_ax)] = jj;
            const Index lin_d = lay_d->grid_linear(d_coord);
            tile::multiply_slice(
                alpha,
                s_tile,
                tiles_d[static_cast<size_t>(lin_d)],
                axis);
        }
    }
}

} // namespace nntile::tensor
