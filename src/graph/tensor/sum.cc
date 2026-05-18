/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum.cc
 * TensorGraph sum operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/sum.hh"
#include "nntile/tensor/sum.hh"

namespace nntile::graph::tensor
{

void TensorSumOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::sum_async (src/tensor/sum.cc).
    const auto& tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tiles_dst.size() != 1)
    {
        throw std::runtime_error(
            "lower_to_tile SUM: scalar output must be one tile");
    }
    constexpr Scalar one = 1.0;
    for(size_t i = 0; i < tiles_src.size(); ++i)
    {
        if(i == 0)
        {
            tile_graph::sum(alpha, tiles_src[i], beta, tiles_dst[0]);
        }
        else
        {
            tile_graph::sum(alpha, tiles_src[i], one, tiles_dst[0]);
        }
    }
}

void sum(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Scalar beta)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sum: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "sum: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sum: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sum: input tensors must have the same dtype");
    }
    if(dst->ndim() != 0)
    {
        throw std::invalid_argument(
            "sum: dst must be a scalar (0-dimensional tensor)");
    }

    auto op = std::make_shared<TensorSumOp>(src, dst, alpha, beta);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
