/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/multiply_fiber_inplace.cc
 * TensorGraph multiply_fiber_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/multiply_fiber_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/multiply_fiber_inplace.hh"
#include "nntile/tensor/multiply_fiber_inplace.hh"

namespace nntile::graph::tensor
{

void TensorMultiplyFiberInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::multiply_fiber_inplace_async
    // (src/tensor/multiply_fiber_inplace.cc).
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile MULTIPLY_FIBER_INPLACE: missing tiling for dst");
    }
    const auto& ts = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& td = tile_lower::tiles_of(ctx.tile_map, dst);
    std::vector<Index> dst_coord;
    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        const Index j = dst_coord[static_cast<size_t>(axis)];
        tile_graph::multiply_fiber_inplace(
            alpha, ts[static_cast<size_t>(j)], td[static_cast<size_t>(lin_d)], axis);
    }
}

void multiply_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_fiber_inplace: input tensors must have the same dtype");
    }
    validate_fiber_shape_and_merge(src, dst, axis, 0,
                                  "multiply_fiber_inplace");

    auto op = std::make_shared<TensorMultiplyFiberInplaceOp>(
        alpha, src, dst, axis);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
