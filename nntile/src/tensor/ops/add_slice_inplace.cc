#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/add_slice_inplace.cc
 * TensorGraph add_slice_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/add_slice_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/add_slice_inplace.hh"

#include "nntile/tile/ops/add_slice_inplace.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"

namespace nntile::tensor
{



void add_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "add_slice_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "add_slice_inplace: src and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(src, dst, axis,
                                            "add_slice_inplace");

    auto op = std::make_shared<TensorAddSliceInplaceOp>(
        src, dst, alpha, beta, axis);
    src->graph()->add_op(op);
}

void TensorAddSliceInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::add_slice_inplace_async (src/tensor/add_slice_inplace.cc).
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ADD_SLICE_INPLACE: missing tiling for src and/or dst");
    }

    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index nd = dst->ndim();
    const Index s_nd = src->ndim();
    const Index s_axis = graph_axis_to_storage(axis, nd);

    std::vector<Index> s_coord;
    std::vector<Index> d_coord(static_cast<size_t>(nd));

    for(Index lin_s = 0; lin_s < lay_s->grid_volume(); ++lin_s)
    {
        lay_s->grid_coord_from_linear(lin_s, s_coord);
        for(Index s = 0; s < nd; ++s)
        {
            if(s == s_axis)
            {
                continue;
            }
            const Index g = storage_axis_to_graph(s, nd);
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
            d_coord[static_cast<size_t>(s)] =
                s_coord[static_cast<size_t>(
                    graph_axis_to_storage(g1, s_nd))];
        }

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(s_axis)];
        for(Index jj = 0; jj < nseg_along_axis; ++jj)
        {
            d_coord[static_cast<size_t>(s_axis)] = jj;
            const Index lin_d = lay_d->grid_linear(d_coord);
            tile::add_slice_inplace(
                alpha,
                tiles_s[static_cast<size_t>(lin_s)],
                beta,
                tiles_d[static_cast<size_t>(lin_d)],
                s_axis);
        }
    }
}

} // namespace nntile::tensor
