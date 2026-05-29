/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/multiply_inplace.cc
 * TensorGraph multiply_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor_graph/ops/multiply_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor_graph/tile_lowering_helpers.hh"
#include "nntile/tile_graph/lowering_context.hh"
#include "nntile/tile_graph/ops/multiply_inplace.hh"
#include "nntile/tensor/multiply_inplace.hh"

namespace nntile::tensor_graph
{



void multiply_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "multiply_inplace: input tensors must have the same dtype");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "multiply_inplace: src and dst must be distinct tensors");
    }
    validate_same_shape_and_merge(src, dst, "multiply_inplace");

    auto op = std::make_shared<TensorMultiplyInplaceOp>(src, dst, alpha);
    src->graph()->add_op(op);
}

void TensorMultiplyInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tiles_src.size() != tiles_dst.size())
    {
        throw std::runtime_error(
            "lower_to_tile MULTIPLY_INPLACE: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(
        src, dst, "MULTIPLY_INPLACE src/dst");
    for(size_t i = 0; i < tiles_src.size(); ++i)
    {
        tile_graph::multiply_inplace(alpha, tiles_src[i], tiles_dst[i]);
    }
}

} // namespace nntile::tensor_graph
