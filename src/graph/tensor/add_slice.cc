/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_slice.cc
 * TensorGraph add_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_slice.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/add_slice.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/tensor/add_slice.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* add_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "add_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must have the same dtype");
    }
    validate_slice_shape_and_merge(src1, src2, axis, "add_slice");

    // Output shape matches src2
    std::vector<Index> output_shape = src2->shape();
    TensorGraph::TensorNode* output = src2->graph()->data(
        std::move(output_shape),
        output_name,
        src2->dtype());
    output->set_axes(src2->axes());

    auto op = std::make_shared<TensorAddSliceOp>(
        src1, src2, output, alpha, beta, axis);
    src1->graph()->add_op(op);

    return output;
}

void add_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "add_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "add_slice: input tensors must have the same dtype");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "add_slice: src1, src2, and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(src1, src2, axis, "add_slice");
    validate_same_shape_and_merge(src2, dst, "add_slice");

    auto op = std::make_shared<TensorAddSliceOp>(
        src1, src2, dst, alpha, beta, axis);
    src1->graph()->add_op(op);
}

void TensorAddSliceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::add_slice_async (src/tensor/add_slice.cc):
    // broadcast src1 along `axis` to each matching dst/src2 tile.
    const TensorAxisLayout* lay_s1 = ctx.tiling.find(src1);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s1 == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ADD_SLICE: missing tiling for src1 and/or dst");
    }

    tile_lower::assert_same_elementwise_layout(src2, dst, "ADD_SLICE src2/dst");

    const auto& tiles_s1 = tile_lower::tiles_of(ctx.tile_map, src1);
    const auto& tiles_s2 = tile_lower::tiles_of(ctx.tile_map, src2);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> s1_coord;
    std::vector<Index> dst_coord(static_cast<size_t>(dst->ndim()));

    for(Index lin_s1 = 0; lin_s1 < lay_s1->grid_volume(); ++lin_s1)
    {
        lay_s1->grid_coord_from_linear(lin_s1, s1_coord);
        TileGraph::TileNode* s1_tile = tiles_s1[static_cast<size_t>(lin_s1)];

        for(Index j = 0; j < axis; ++j)
        {
            dst_coord[static_cast<size_t>(j)] =
                s1_coord[static_cast<size_t>(j)];
        }
        for(Index j = axis + 1; j < dst->ndim(); ++j)
        {
            dst_coord[static_cast<size_t>(j)] =
                s1_coord[static_cast<size_t>(j - 1)];
        }

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(axis)];
        for(Index jj = 0; jj < nseg_along_axis; ++jj)
        {
            dst_coord[static_cast<size_t>(axis)] = jj;
            const Index lin_d = lay_d->grid_linear(dst_coord);
            tile_graph::add_slice(
                alpha,
                s1_tile,
                beta,
                tiles_s2[static_cast<size_t>(lin_d)],
                tiles_d[static_cast<size_t>(lin_d)],
                axis);
        }
    }
}

} // namespace nntile::graph::tensor
