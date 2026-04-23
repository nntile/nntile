/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/subtract_indexed_outputs.cc
 * TensorGraph subtract_indexed_outputs operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/subtract_indexed_outputs.hh"

#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/subtract_indexed_outputs.hh"

namespace nntile::graph::tensor
{

void TensorSubtractIndexedOutputsOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::subtract_indexed_outputs_async
    // (src/tensor/subtract_indexed_outputs.cc): one tile op per dst grid cell,
    // paired with the labels tile at grid coords (dst_coord[1], ..., dst_coord[ndim-1]).
    // That equals shared linear index i with labels.get_tile(i) when dst axis 0 has
    // a single tile (StarPU tensor requires shape[0] == basetile_shape[0]).
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    const TensorAxisLayout* lay_lab = ctx.tiling.find(labels);
    if(lay_d == nullptr || lay_lab == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SUBTRACT_INDEXED_OUTPUTS: missing tiling for dst "
            "and/or labels");
    }
    const auto& t_lab = tile_lower::tiles_of(ctx.tile_map, labels);
    const auto& t_dst = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> dst_coord;
    std::vector<Index> lab_coord(static_cast<size_t>(labels->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        for(Index j = 0; j < labels->ndim(); ++j)
        {
            lab_coord[static_cast<size_t>(j)] =
                dst_coord[static_cast<size_t>(j + 1)];
        }
        const Index lin_l = lay_lab->grid_linear(lab_coord);
        tile_graph::subtract_indexed_outputs(val,
            t_lab[static_cast<size_t>(lin_l)],
            t_dst[static_cast<size_t>(lin_d)], ignore_index);
    }
}

void subtract_indexed_outputs(Scalar val,
                             TensorGraph::TensorNode* labels,
                             TensorGraph::TensorNode* dst,
                             Index ignore_index)
{
    if(labels == nullptr || dst == nullptr)
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must be non-null");
    if(labels->graph() != dst->graph())
        throw std::invalid_argument(
            "subtract_indexed_outputs: tensors must belong to same graph");
    if(labels->dtype() != DataType::INT64)
        throw std::invalid_argument(
            "subtract_indexed_outputs: labels must have INT64 dtype");
    // labels.dim[i] == dst.dim[i+1]: labels index the batch dims of dst
    if(labels->ndim() + 1 != dst->ndim())
    {
        throw std::invalid_argument(
            "subtract_indexed_outputs: dst must have ndim = labels.ndim + 1");
    }
    for(Index i = 0; i < labels->ndim(); ++i)
    {
        if(labels->shape()[i] != dst->shape()[i + 1])
        {
            throw std::invalid_argument(
                "subtract_indexed_outputs: labels.dim[" +
                std::to_string(i) + "] must match dst.dim[" +
                std::to_string(i + 1) + "] (" +
                std::to_string(labels->shape()[i]) + " vs " +
                std::to_string(dst->shape()[i + 1]) + ")");
        }
        merge_axis(labels->mutable_axes()[i],
                   dst->mutable_axes()[i + 1]);
    }

    auto op = std::make_shared<TensorSubtractIndexedOutputsOp>(
        val, labels, dst, ignore_index);
    dst->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
