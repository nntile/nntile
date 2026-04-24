/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/hypot_inplace.cc
 * TensorGraph hypot_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/hypot_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/hypot_inplace.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/hypot_inplace.hh"

namespace nntile::graph::tensor
{

void TensorHypotInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vs = tile_lower::tiles_of(m, src);
    const auto& vd = tile_lower::tiles_of(m, dst);
    if(vs.size() != vd.size())
    {
        throw std::runtime_error(
            "lower_to_tile HYPOT_INPLACE: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(src, dst, "HYPOT_INPLACE");
    for(size_t i = 0; i < vs.size(); ++i)
    {
        tile_graph::hypot_inplace(alpha, vs[i], beta, vd[i]);
    }
}

void hypot_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "hypot_inplace: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "hypot_inplace: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "hypot_inplace: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "hypot_inplace: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src, dst, "hypot_inplace");

    auto op = std::make_shared<TensorHypotInplaceOp>(
        alpha, beta, src, dst);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
