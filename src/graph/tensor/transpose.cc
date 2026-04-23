/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/transpose.cc
 * TensorGraph transpose operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/transpose.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/transpose.hh"
#include "nntile/tensor/transpose.hh"
#include "nntile/tile/traits.hh"

namespace nntile::graph::tensor
{

namespace
{

//! Tensor tile grids use Fortran-order linear indices (dim 0 varies fastest);
//! TensorAxisLayout / tile_map use `grid_linear` (dim 0 slowest). Convert
//! between the two for a given axis layout.
Index fortran_tile_linear_to_layout_linear(
    Index fort_lin, const TensorAxisLayout& lay)
{
    const auto& gsh = lay.grid_shape();
    const Index nd = static_cast<Index>(gsh.size());
    std::vector<Index> coord(static_cast<size_t>(nd), 0);
    Index rem = fort_lin;
    for(Index d = 0; d < nd; ++d)
    {
        const Index gs = gsh[static_cast<size_t>(d)];
        coord[static_cast<size_t>(d)] = rem % gs;
        rem /= gs;
    }
    return lay.grid_linear(coord);
}

} // namespace

void TensorTransposeOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::transpose_async (src/tensor/transpose.cc).
    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile TRANSPOSE: missing tiling for src or dst");
    }
    const nntile::tile::TileTraits grid_src(lay_s->grid_shape());
    const Index grid_m = grid_src.matrix_shape[static_cast<size_t>(ndim)][0];
    const Index grid_n = grid_src.matrix_shape[static_cast<size_t>(ndim)][1];
    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);
    for(Index j = 0; j < grid_n; ++j)
    {
        for(Index i = 0; i < grid_m; ++i)
        {
            const Index lin_src_f = i + j * grid_m;
            const Index lin_dst_f = i * grid_n + j;
            const Index lin_s =
                fortran_tile_linear_to_layout_linear(lin_src_f, *lay_s);
            const Index lin_d =
                fortran_tile_linear_to_layout_linear(lin_dst_f, *lay_d);
            tile_graph::transpose(
                alpha,
                tiles_s[static_cast<size_t>(lin_s)],
                tiles_d[static_cast<size_t>(lin_d)],
                ndim);
        }
    }
}

TensorGraph::TensorNode* transpose(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index ndim)
{
    if(src == nullptr)
        throw std::invalid_argument("transpose: input tensor must be non-null");
    if(ndim <= 0 || ndim >= src->ndim())
        throw std::invalid_argument("transpose: ndim must be in (0, src.ndim)");
    std::vector<Index> src_shape = src->shape();
    Index n = src->ndim();
    std::vector<Index> output_shape(n);
    for(Index i = 0; i < n; ++i)
        output_shape[i] = src_shape[(i + ndim) % n];
    TensorGraph::TensorNode* output = src->graph()->data(
        std::move(output_shape), output_name, src->dtype());
    for(Index i = 0; i < n; ++i)
    {
        merge_axis(src->mutable_axes()[(i + ndim) % n],
                   output->mutable_axes()[i]);
    }
    auto op = std::make_shared<TensorTransposeOp>(src, output, ndim, alpha);
    src->graph()->add_op(op);
    return output;
}

void transpose(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index ndim)
{
    if(src == nullptr || dst == nullptr)
        throw std::invalid_argument("transpose: tensors must be non-null");
    if(src == dst)
        throw std::invalid_argument("transpose: src and dst must be distinct tensors");
    if(src->graph() != dst->graph())
        throw std::invalid_argument("transpose: tensors must belong to same graph");
    if(src->dtype() != dst->dtype())
        throw std::invalid_argument("transpose: tensors must have same dtype");
    if(ndim <= 0 || ndim >= src->ndim())
        throw std::invalid_argument("transpose: ndim must be in (0, src.ndim)");
    Index n = src->ndim();
    const auto& src_shape = src->shape();
    const auto& dst_shape = dst->shape();
    if(dst_shape.size() != n)
    {
        throw std::invalid_argument(
            "transpose: dst.ndim must equal src.ndim (" +
            std::to_string(dst_shape.size()) + " vs " + std::to_string(n) + ")");
    }
    for(Index i = 0; i < n; ++i)
    {
        Index expected = src_shape[(i + ndim) % n];
        if(dst_shape[i] != expected)
        {
            throw std::invalid_argument(
                "transpose: dst shape must be permuted src shape; mismatch at "
                "dimension " + std::to_string(i) + " (" +
                std::to_string(dst_shape[i]) + " vs " +
                std::to_string(expected) + ")");
        }
    }
    for(Index i = 0; i < n; ++i)
    {
        merge_axis(src->mutable_axes()[(i + ndim) % n],
                   dst->mutable_axes()[i]);
    }
    auto op = std::make_shared<TensorTransposeOp>(src, dst, ndim, alpha);
    src->graph()->add_op(op);
}

} // namespace nntile::graph::tensor
