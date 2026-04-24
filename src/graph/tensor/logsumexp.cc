/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/logsumexp.cc
 * TensorGraph logsumexp operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/logsumexp.hh"

#include <stdexcept>
#include <utility>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/logsumexp.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/logsumexp.hh"

namespace nntile::graph::tensor
{

void TensorLogsumexpOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::logsumexp_async (src/tensor/logsumexp.cc).
    const TensorAxisLayout* lay_src = ctx.tiling.find(src);
    const TensorAxisLayout* lay_dst = ctx.tiling.find(dst);
    if(lay_src == nullptr || lay_dst == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile LOGSUMEXP: missing tiling for src and/or dst");
    }
    if(lay_src->grid_volume() != lay_dst->grid_volume())
    {
        throw std::runtime_error(
            "lower_to_tile LOGSUMEXP: src/dst grid volume mismatch");
    }
    const auto& tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);
    if(tiles_src.size() != tiles_dst.size())
    {
        throw std::runtime_error(
            "lower_to_tile LOGSUMEXP: tile count mismatch");
    }
    for(size_t i = 0; i < tiles_src.size(); ++i)
    {
        tile_graph::logsumexp(tiles_src[i], tiles_dst[i]);
    }
}

TensorGraph::TensorNode* logsumexp(
    TensorGraph::TensorNode* src,
    const std::string& output_name)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "logsumexp: input tensor must be non-null");
    }

    // dst shape: src.shape[1:] (src has shape[0]=2 for maxsumexp format)
    std::vector<Index> output_shape;
    if(src->ndim() > 1)
    {
        output_shape.assign(src->shape().begin() + 1, src->shape().end());
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        std::move(output_shape),
        output_name,
        src->dtype());

    logsumexp(src, dst);

    return dst;
}

void logsumexp(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "logsumexp: input tensors must be non-null");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "logsumexp: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "logsumexp: input tensors must have the same dtype");
    }
    validate_logsumexp_shape_and_merge(src, dst, "logsumexp");

    auto op = std::make_shared<TensorLogsumexpOp>(src, dst);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
