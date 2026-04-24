/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/softmax.cc
 * TensorGraph softmax operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/softmax.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/softmax.hh"
#include "nntile/tensor/softmax.hh"

namespace nntile::graph::tensor
{

void TensorSoftmaxOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::softmax_async (src/tensor/softmax.cc); tile pairing
    // mirrors TensorSoftmaxInplaceOp::lower_to_tile (softmax_inplace.cc).
    const TensorAxisLayout* lay_m = ctx.tiling.find(maxsumexp);
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_m == nullptr || lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SOFTMAX: missing tiling for maxsumexp, src, "
            "and/or dst");
    }

    const auto& tiles_m = tile_lower::tiles_of(ctx.tile_map, maxsumexp);
    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> m_coord;
    std::vector<Index> coord(static_cast<size_t>(dst->ndim()));

    for(Index lin_m = 0; lin_m < lay_m->grid_volume(); ++lin_m)
    {
        lay_m->grid_coord_from_linear(lin_m, m_coord);
        TileGraph::TileNode* m_tile = tiles_m[static_cast<size_t>(lin_m)];

        for(Index j = 0; j < axis; ++j)
        {
            coord[static_cast<size_t>(j)] =
                m_coord[static_cast<size_t>(j + 1)];
        }
        for(Index j = axis + 1; j < dst->ndim(); ++j)
        {
            coord[static_cast<size_t>(j)] =
                m_coord[static_cast<size_t>(j)];
        }

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(axis)];
        for(Index j = 0; j < nseg_along_axis; ++j)
        {
            coord[static_cast<size_t>(axis)] = j;
            const Index lin = lay_d->grid_linear(coord);
            TileGraph::TileNode* s_tile = tiles_s[static_cast<size_t>(lin)];
            TileGraph::TileNode* d_tile = tiles_d[static_cast<size_t>(lin)];
            tile_graph::softmax(m_tile, s_tile, alpha, d_tile, axis);
        }
    }
}

TensorGraph::TensorNode* softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || src == nullptr)
    {
        throw std::invalid_argument(
            "softmax: input tensors must be non-null");
    }
    if(maxsumexp->graph() != src->graph())
    {
        throw std::invalid_argument(
            "softmax: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "softmax: input tensors must have the same dtype");
    }
    // maxsumexp has shape with 2 at axis, src has full shape

    TensorGraph::TensorNode* dst = src->graph()->data(
        src->shape(),
        output_name,
        src->dtype());
    dst->set_axes(src->axes());

    softmax(maxsumexp, src, dst, alpha, axis);

    return dst;
}

void softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "softmax: input tensors must be non-null");
    }
    if(maxsumexp->graph() != src->graph() || maxsumexp->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "softmax: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != src->dtype() || maxsumexp->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "softmax: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src, dst, "softmax");
    validate_maxsumexp_shape_and_merge(src, maxsumexp, axis, "softmax");

    auto op = std::make_shared<TensorSoftmaxOp>(
        maxsumexp, src, dst, alpha, axis);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
