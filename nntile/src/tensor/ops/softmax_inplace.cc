#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/softmax_inplace.cc
 * TensorGraph softmax_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/softmax_inplace.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/ops/softmax_inplace.hh"
#include "nntile/tensor/ops/softmax_inplace.hh"

namespace nntile::tensor
{

namespace
{

void maxsumexp_to_src_storage_coord(const std::vector<Index> &m_storage,
    Index axis,
    Index src_ndim,
    std::vector<Index> &src_storage)
{
    src_storage.resize(static_cast<size_t>(src_ndim));
    for (Index s = 0; s < src_ndim; ++s)
    {
        const Index g = storage_axis_to_graph(s, src_ndim);
        if (g == axis)
        {
            continue;
        }
        const Index m_g = (g < axis) ? g : (g - 1);
        const Index dst_ndim = static_cast<Index>(m_storage.size());
        const Index m_s = graph_axis_to_storage(m_g, dst_ndim);
        src_storage[static_cast<size_t>(s)] = m_storage[static_cast<size_t>(m_s)];
    }
}

} // namespace

void softmax_inplace(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis)
{
    if(maxsumexp == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must be non-null");
    }
    if(maxsumexp->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must belong to the same graph");
    }
    if(maxsumexp->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "softmax_inplace: input tensors must have the same dtype");
    }
    // maxsumexp has shape with 2 at axis, dst has full shape

    auto op = std::make_shared<TensorSoftmaxInplaceOp>(
        maxsumexp, dst, alpha, axis);
    maxsumexp->graph()->add_op(op);
}

void TensorSoftmaxInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::softmax_inplace_async: iterate maxsumexp tiles,
    // then all dst tiles along `axis` sharing that fiber (see
    // src/tensor/softmax_inplace.cc and src/tensor/softmax.cc).
    const TensorAxisLayout* lay_m = ctx.tiling.find(maxsumexp);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_m == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SOFTMAX_INPLACE: missing tiling for maxsumexp "
            "and/or dst");
    }

    const auto& tiles_m = tile_lower::tiles_of(ctx.tile_map, maxsumexp);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index nd = dst->ndim();
    const Index s_axis = graph_axis_to_storage(axis, nd);

    std::vector<Index> m_coord;
    std::vector<Index> dst_coord(static_cast<size_t>(nd));

    for(Index lin_m = 0; lin_m < lay_m->grid_volume(); ++lin_m)
    {
        lay_m->grid_coord_from_linear(lin_m, m_coord);
        TileGraph::TileNode* m_tile = tiles_m[static_cast<size_t>(lin_m)];

        maxsumexp_to_src_storage_coord(m_coord, axis, nd, dst_coord);

        const Index nseg_along_axis =
            lay_d->grid_shape()[static_cast<size_t>(s_axis)];
        for(Index j = 0; j < nseg_along_axis; ++j)
        {
            dst_coord[static_cast<size_t>(s_axis)] = j;
            const Index lin_d = lay_d->grid_linear(dst_coord);
            TileGraph::TileNode* d_tile =
                tiles_d[static_cast<size_t>(lin_d)];
            tile::softmax_inplace(m_tile, alpha, d_tile, s_axis);
        }
    }
}

} // namespace nntile::tensor
