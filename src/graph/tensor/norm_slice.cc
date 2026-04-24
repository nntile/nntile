/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/norm_slice.cc
 * TensorGraph norm_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/norm_slice.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/norm_slice.hh"

#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/norm_slice.hh"
#include "nntile/graph/tile/norm_slice_inplace.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* norm_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis,
    int redux)
{
    if(src1 == nullptr || src2 == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must have the same dtype");
    }

    std::vector<Index> output_shape = src2->shape();
    TensorGraph::TensorNode* dst = src1->graph()->data(
        std::move(output_shape),
        output_name,
        src1->dtype());

    validate_slice_shape_and_merge(dst, src1, axis, "norm_slice");
    validate_same_shape_and_merge(src2, dst, "norm_slice");

    auto op = std::make_shared<TensorNormSliceOp>(
        alpha, beta, src1, src2, dst, axis, redux);
    src1->graph()->add_op(op);

    return dst;
}

void norm_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "norm_slice: input tensors must have the same dtype");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "norm_slice: src1, src2, and dst must be distinct tensors");
    }
    validate_slice_shape_and_merge(dst, src1, axis, "norm_slice");
    validate_same_shape_and_merge(src2, dst, "norm_slice");

    auto op = std::make_shared<TensorNormSliceOp>(
        alpha, beta, src1, src2, dst, axis, redux);
    src1->graph()->add_op(op);
}

void TensorNormSliceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::norm_slice_async (src/tensor/norm_slice.cc).
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    const TensorAxisLayout* lay_s1 = ctx.tiling.find(src1);
    if(lay_d == nullptr || lay_s1 == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile NORM_SLICE: missing tiling for dst and/or src1");
    }

    tile_lower::assert_same_elementwise_layout(src2, dst, "NORM_SLICE src2/dst");

    const auto& tiles_s1 = tile_lower::tiles_of(ctx.tile_map, src1);
    const auto& tiles_s2 = tile_lower::tiles_of(ctx.tile_map, src2);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    constexpr Scalar one = 1.0;
    std::vector<Index> dst_coord;
    std::vector<Index> s1_coord(static_cast<size_t>(src1->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        for(Index j = 0, k = 0; j < src1->ndim(); ++j)
        {
            if(j == axis)
            {
                continue;
            }
            s1_coord[static_cast<size_t>(j)] = dst_coord[static_cast<size_t>(k)];
            ++k;
        }

        const Index nseg_along_axis =
            lay_s1->grid_shape()[static_cast<size_t>(axis)];
        for(Index jj = 0; jj < nseg_along_axis; ++jj)
        {
            s1_coord[static_cast<size_t>(axis)] = jj;
            const Index lin_s1 = lay_s1->grid_linear(s1_coord);
            if(jj == 0)
            {
                tile_graph::norm_slice(
                    alpha,
                    tiles_s1[static_cast<size_t>(lin_s1)],
                    beta,
                    tiles_s2[static_cast<size_t>(lin_d)],
                    tiles_d[static_cast<size_t>(lin_d)],
                    axis,
                    redux);
            }
            else
            {
                tile_graph::norm_slice_inplace(
                    alpha,
                    tiles_s1[static_cast<size_t>(lin_s1)],
                    one,
                    tiles_d[static_cast<size_t>(lin_d)],
                    axis,
                    redux);
            }
        }
    }
}

} // namespace nntile::graph::tensor
