/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/conv2d_bwd_weight_inplace.cc
 * TensorGraph conv2d_bwd_weight_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/conv2d_bwd_weight_inplace.hh"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/clear.hh"
#include "nntile/graph/tile/conv2d_bwd_weight_inplace.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/scale_inplace.hh"
#include "nntile/tensor/conv2d_bwd_weight_inplace.hh"

namespace nntile::graph::tensor
{

namespace
{

Index uniform_extent(const nntile::graph::TensorAxisLayout& lay, Index dim,
    const char* op)
{
    const auto& gs = lay.grid_shape();
    if(dim < 0 || dim >= static_cast<Index>(gs.size()))
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": conv uniform_extent: bad dim");
    }
    Index first = -1;
    std::vector<Index> coord(gs.size(), 0);
    for(Index lin = 0; lin < lay.grid_volume(); ++lin)
    {
        lay.grid_coord_from_linear(lin, coord);
        const Index ext = lay.tile_shape_at(coord)[static_cast<size_t>(dim)];
        if(first < 0)
        {
            first = ext;
        }
        else if(ext != first)
        {
            throw std::runtime_error(std::string("lower_to_tile ") + op +
                ": conv requires uniform tile extent per spatial/batch axis");
        }
    }
    return first;
}

void assert_full_in_channels(
    const nntile::graph::TensorAxisLayout& lay, const char* op)
{
    if(lay.grid_shape().size() < 4 || lay.grid_shape()[2] != 1)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": channel axis must be a single tile (full C)");
    }
    const Index ext = uniform_extent(lay, 2, op);
    if(ext != lay.tensor_shape()[2])
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": channel tile must cover full channel extent");
    }
}

} // namespace



void conv2d_bwd_weight_inplace(Scalar alpha,
                               TensorGraph::TensorNode* X,
                               TensorGraph::TensorNode* dY,
                               Scalar beta,
                               TensorGraph::TensorNode* dC,
                               std::array<Index, 2> padding,
                               std::array<Index, 2> stride,
                               std::array<Index, 2> dilation)
{
    if(X == nullptr || dY == nullptr || dC == nullptr)
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: tensors must be non-null");
    if(X->graph() != dY->graph() || dY->graph() != dC->graph())
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: tensors must belong to same graph");
    if(X->dtype() != dY->dtype() || dY->dtype() != dC->dtype())
        throw std::invalid_argument(
            "conv2d_bwd_weight_inplace: tensors must have same dtype");
    auto op = std::make_shared<TensorConv2dBwdWeightInplaceOp>(
        alpha, X, dY, beta, dC, padding, stride, dilation);
    dC->graph()->add_op(op);
}

void TensorConv2dBwdWeightInplaceOp::lower_to_tile(
    const LoweringContext& ctx) const
{
    constexpr const char* op = "CONV2D_BWD_WEIGHT_INPLACE";
    const TensorAxisLayout* lay_x = ctx.tiling.find(X);
    const TensorAxisLayout* lay_dy = ctx.tiling.find(dY);
    const TensorAxisLayout* lay_dc = ctx.tiling.find(dC);
    if(lay_x == nullptr || lay_dy == nullptr || lay_dc == nullptr)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": missing tiling for X/dY/dC");
    }
    if(lay_dc->grid_volume() != 1)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": dC must be a single tile");
    }

    assert_full_in_channels(*lay_x, op);
    assert_full_in_channels(*lay_dy, op);

    const Index x_bs0 = uniform_extent(*lay_x, 0, op);
    const Index x_bs1 = uniform_extent(*lay_x, 1, op);
    const Index x_bs3 = uniform_extent(*lay_x, 3, op);
    const Index dy_bs0 = uniform_extent(*lay_dy, 0, op);
    const Index dy_bs1 = uniform_extent(*lay_dy, 1, op);
    const Index dy_bs3 = uniform_extent(*lay_dy, 3, op);
    if(dy_bs3 != x_bs3)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": dY and X batch tile extents must match");
    }

    const auto& tiles_x = tile_lower::tiles_of(ctx.tile_map, X);
    const auto& tiles_dy = tile_lower::tiles_of(ctx.tile_map, dY);
    const auto& tiles_dc = tile_lower::tiles_of(ctx.tile_map, dC);

    TileGraph::TileNode* dc_tile = tiles_dc[0];
    std::vector<Index> dc_zero(lay_dc->grid_shape().size(), 0);
    const auto dc_ts = lay_dc->tile_shape_at(dc_zero);

    const Index Kx = dC->shape()[0];
    const Index Ky = dC->shape()[1];

    std::vector<Index> x_coord(4);
    std::vector<Index> dy_coord(4);

    const Index gdy0 = lay_dy->grid_shape()[0];
    const Index gdy1 = lay_dy->grid_shape()[1];

    Scalar dc_tile_beta = beta;
    bool initialized = false;

    for(Index lin_x = 0; lin_x < lay_x->grid_volume(); ++lin_x)
    {
        lay_x->grid_coord_from_linear(lin_x, x_coord);
        const auto x_ts = lay_x->tile_shape_at(x_coord);
        TileGraph::TileNode* x_tile = tiles_x[static_cast<size_t>(lin_x)];

        Index x_lo_m = 0, x_hi_m = 0;
        Index x_lo_n = 0, x_hi_n = 0;
        lay_x->tile_axis_global_range(x_coord, 0, x_lo_m, x_hi_m);
        lay_x->tile_axis_global_range(x_coord, 1, x_lo_n, x_hi_n);
        const Index X_start_m = x_lo_m;
        const Index X_end_m = x_hi_m + 1;
        const Index X_start_n = x_lo_n;
        const Index X_end_n = x_hi_n + 1;

        Index dY_start_m = (X_start_m + padding[0] - dilation[0] * (Kx - 1)
                               + stride[0] - 1)
            / stride[0];
        Index dY_end_m = (X_end_m - 1 + padding[0] + stride[0]) / stride[0];
        Index dY_start_n = (X_start_n + padding[1] - dilation[1] * (Ky - 1)
                               + stride[1] - 1)
            / stride[1];
        Index dY_end_n = (X_end_n - 1 + padding[1] + stride[1]) / stride[1];

        Index dY_start_tile_m = dY_start_m / dy_bs0;
        Index dY_end_tile_m = (dY_end_m - 1) / dy_bs0 + 1;
        Index dY_start_tile_n = dY_start_n / dy_bs1;
        Index dY_end_tile_n = (dY_end_n - 1) / dy_bs1 + 1;

        if(dY_end_tile_m <= 0 || dY_start_tile_m >= gdy0 || dY_end_tile_n <= 0
            || dY_start_tile_n >= gdy1)
        {
            continue;
        }

        dy_coord[2] = x_coord[2];
        dy_coord[3] = x_coord[3];
        const Index start_m = std::max(dY_start_tile_m, Index(0));
        const Index end_m = std::min(dY_end_tile_m, gdy0);
        const Index start_n = std::max(dY_start_tile_n, Index(0));
        const Index end_n = std::min(dY_end_tile_n, gdy1);

        for(Index dY_i = start_m; dY_i < end_m; ++dY_i)
        {
            dy_coord[0] = dY_i;
            for(Index dY_j = start_n; dY_j < end_n; ++dY_j)
            {
                dy_coord[1] = dY_j;
                const Index lin_dy = lay_dy->grid_linear(dy_coord);
                const auto dy_ts = lay_dy->tile_shape_at(dy_coord);
                const Index offset_m =
                    X_start_m + padding[0] - stride[0] * dY_i * dy_bs0;
                const Index offset_n =
                    X_start_n + padding[1] - stride[1] * dY_j * dy_bs1;
                tile_graph::conv2d_bwd_weight_inplace(
                    x_ts[0],
                    x_ts[1],
                    x_ts[2],
                    x_ts[3],
                    dy_ts[0],
                    dy_ts[1],
                    stride[0],
                    stride[1],
                    dy_ts[2],
                    offset_m,
                    offset_n,
                    alpha,
                    x_tile,
                    tiles_dy[static_cast<size_t>(lin_dy)],
                    dc_ts[0],
                    dc_ts[1],
                    dilation[0],
                    dilation[1],
                    dc_tile_beta,
                    dc_tile);
                dc_tile_beta = 1.0;
                initialized = true;
            }
        }
    }

    if(!initialized)
    {
        if(beta == 0.0)
        {
            tile_graph::clear(dc_tile);
        }
        else if(beta != 1.0)
        {
            tile_graph::scale_inplace(beta, dc_tile);
        }
    }
}

} // namespace nntile::graph::tensor
