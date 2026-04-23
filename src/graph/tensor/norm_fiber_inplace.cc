/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm_fiber_inplace.cc
 * TensorGraph norm_fiber_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm_fiber_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/norm_fiber_inplace.hh"
#include "nntile/tensor/norm_fiber_inplace.hh"

namespace nntile::graph::tensor
{

void TensorNormFiberInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::norm_fiber_inplace_async (src/tensor/norm_fiber_inplace.cc).
    const TensorAxisLayout* lay1 = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay1 == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile NORM_FIBER_INPLACE: missing tiling for src and/or "
            "dst");
    }
    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);
    constexpr Scalar one = 1.0;
    std::vector<Index> s1_coord;
    std::vector<Index> dst_coord(static_cast<size_t>(dst->ndim()));
    const Index fiber_prefix = src->ndim() - batch_ndim;

    for(Index lin1 = 0; lin1 < lay1->grid_volume(); ++lin1)
    {
        lay1->grid_coord_from_linear(lin1, s1_coord);
        bool init_first = true;
        for(Index j = 0; j < fiber_prefix; ++j)
        {
            if(j != axis && s1_coord[static_cast<size_t>(j)] != 0)
            {
                init_first = false;
                break;
            }
        }
        dst_coord[0] = s1_coord[static_cast<size_t>(axis)];
        for(Index b = 0; b < batch_ndim; ++b)
        {
            dst_coord[static_cast<size_t>(b + 1)] =
                s1_coord[static_cast<size_t>(src->ndim() - batch_ndim + b)];
        }
        const Index lin_d = lay_d->grid_linear(dst_coord);
        if(init_first)
        {
            tile_graph::norm_fiber_inplace(
                alpha,
                tiles_s[static_cast<size_t>(lin1)],
                beta,
                tiles_d[static_cast<size_t>(lin_d)],
                axis,
                batch_ndim,
                redux);
        }
        else
        {
            tile_graph::norm_fiber_inplace(
                alpha,
                tiles_s[static_cast<size_t>(lin1)],
                one,
                tiles_d[static_cast<size_t>(lin_d)],
                axis,
                batch_ndim,
                redux);
        }
    }
}

void norm_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim,
    int redux)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "norm_fiber_inplace: src and dst must be distinct tensors");
    }
    validate_fiber_shape_and_merge(dst, src, axis, batch_ndim,
                                   "norm_fiber_inplace");

    auto op = std::make_shared<TensorNormFiberInplaceOp>(
        alpha, beta, src, dst, axis, batch_ndim, redux);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
