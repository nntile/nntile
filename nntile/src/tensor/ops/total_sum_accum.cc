#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/total_sum_accum.cc
 * TensorGraph total_sum_accum operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/total_sum_accum.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/total_sum_accum.hh"
#include "nntile/tensor/ops/total_sum_accum.hh"

namespace nntile::tensor
{

void TensorTotalSumAccumOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::total_sum_accum_async (src/tensor/total_sum_accum.cc).
    const TensorAxisLayout* lay_l = ctx.tiling.find(class_labels);
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_e = ctx.tiling.find(logsumexp);
    if(lay_l == nullptr || lay_s == nullptr || lay_e == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile TOTAL_SUM_ACCUM: missing tiling for class_labels, "
            "src, and/or logsumexp");
    }
    const auto& t_lse = tile_lower::tiles_of(ctx.tile_map, logsumexp);
    const auto& t_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& t_lab = tile_lower::tiles_of(ctx.tile_map, class_labels);
    const auto& t_val = tile_lower::tiles_of(ctx.tile_map, val);
    if(t_val.size() != 1)
    {
        throw std::runtime_error(
            "lower_to_tile TOTAL_SUM_ACCUM: val must be single-tile");
    }
    const bool trailing_class =
        class_labels->shape()[0] == src->shape()[0];
    const Index spatial_offset = trailing_class ? 0 : 1;

    std::vector<Index> src_coord;
    std::vector<Index> lab_coord(static_cast<size_t>(class_labels->ndim()));
    for(Index lin_s = 0; lin_s < lay_s->grid_volume(); ++lin_s)
    {
        lay_s->grid_coord_from_linear(lin_s, src_coord);
        for(Index j = 0; j < class_labels->ndim(); ++j)
        {
            lab_coord[static_cast<size_t>(j)] =
                src_coord[static_cast<size_t>(j + spatial_offset)];
        }
        const Index lin_l = lay_l->grid_linear(lab_coord);
        tile::total_sum_accum(
            alpha,
            t_lse[static_cast<size_t>(lin_l)],
            t_src[static_cast<size_t>(lin_s)],
            t_lab[static_cast<size_t>(lin_l)],
            t_val[0],
            ignore_index);
    }
}

void total_sum_accum(
    Scalar alpha,
    TensorGraph::TensorNode* logsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* class_labels,
    TensorGraph::TensorNode* val,
    Index ignore_index)
{
    if(logsumexp == nullptr || src == nullptr || class_labels == nullptr ||
       val == nullptr)
    {
        throw std::invalid_argument(
            "total_sum_accum: input tensors must be non-null");
    }
    if(logsumexp->graph() != src->graph() ||
       logsumexp->graph() != class_labels->graph() ||
       logsumexp->graph() != val->graph())
    {
        throw std::invalid_argument(
            "total_sum_accum: input tensors must belong to the same graph");
    }
    if(logsumexp->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "total_sum_accum: logsumexp and src must have the same dtype");
    }
    if(class_labels->dtype() != DataType::INT64)
    {
        throw std::invalid_argument(
            "total_sum_accum: class_labels must have INT64 dtype");
    }
    if(val->dtype() != DataType::FP32)
    {
        throw std::invalid_argument(
            "total_sum_accum: val must have FP32 dtype");
    }
    validate_same_shape_and_merge(logsumexp, class_labels, "total_sum_accum");
    validate_logsumexp_shape_and_merge(src, logsumexp, "total_sum_accum");

    auto op = std::make_shared<TensorTotalSumAccumOp>(
        alpha, logsumexp, src, class_labels, val, ignore_index);
    logsumexp->graph()->add_op(op);
}

} // namespace nntile::tensor
