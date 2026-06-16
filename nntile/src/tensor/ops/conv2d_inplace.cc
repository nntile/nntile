#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/conv2d_inplace.cc
 * TensorGraph conv2d_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/conv2d_inplace.hh"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/ops/clear.hh"
#include "nntile/tile/ops/conv2d_inplace.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/scale_inplace.hh"
#include "nntile/tensor/ops/conv2d_inplace.hh"

namespace nntile::tensor
{

namespace
{

// Graph WHCN labels [W, H, C, N]; layout coords are storage axes.
constexpr Index g_WHCN = 4;
const Index s_W = graph_axis_to_storage(0, g_WHCN);
const Index s_H = graph_axis_to_storage(1, g_WHCN);
const Index s_C = graph_axis_to_storage(2, g_WHCN);
const Index s_N = graph_axis_to_storage(3, g_WHCN);

Index uniform_extent(const nntile::TensorAxisLayout& lay, Index dim,
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
    const nntile::TensorAxisLayout& lay, const char* op)
{
    if(lay.grid_shape().size() < g_WHCN || lay.grid_shape()[s_C] != 1)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": channel axis must be a single tile (full C_in)");
    }
    const Index ext = uniform_extent(lay, s_C, op);
    if(ext != lay.tensor_shape()[s_C])
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": channel tile must cover full C_in / C_out");
    }
}

} // namespace



void conv2d_inplace(Scalar alpha,
                    TensorGraph::TensorNode* X,
                    TensorGraph::TensorNode* C,
                    Scalar beta,
                    TensorGraph::TensorNode* Y,
                    std::array<Index, 2> padding,
                    std::array<Index, 2> stride,
                    std::array<Index, 2> dilation)
{
    if(X == nullptr || C == nullptr || Y == nullptr)
        throw std::invalid_argument("conv2d_inplace: tensors must be non-null");
    if(X->graph() != C->graph() || C->graph() != Y->graph())
        throw std::invalid_argument(
            "conv2d_inplace: tensors must belong to same graph");
    if(X->dtype() != C->dtype() || C->dtype() != Y->dtype())
        throw std::invalid_argument(
            "conv2d_inplace: tensors must have same dtype");
    auto op = std::make_shared<TensorConv2dInplaceOp>(
        alpha, X, C, beta, Y, padding, stride, dilation);
    Y->graph()->add_op(op);
}

void TensorConv2dInplaceOp::lower_to_tile(const LoweringContext& ctx) const
{
    constexpr const char* op = "CONV2D_INPLACE";
    const TensorAxisLayout* lay_x = ctx.tiling.find(X);
    const TensorAxisLayout* lay_c = ctx.tiling.find(C);
    const TensorAxisLayout* lay_y = ctx.tiling.find(Y);
    if(lay_x == nullptr || lay_c == nullptr || lay_y == nullptr)
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op + ": missing tiling for X/C/Y");
    }
    if(lay_c->grid_volume() != 1)
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": kernel C must be a single tile (matches tensor API)");
    }

    assert_full_in_channels(*lay_x, op);
    assert_full_in_channels(*lay_y, op);

    const Index x_bs0 = uniform_extent(*lay_x, s_W, op);
    const Index x_bs1 = uniform_extent(*lay_x, s_H, op);
    const Index x_bs3 = uniform_extent(*lay_x, s_N, op);
    const Index y_bs0 = uniform_extent(*lay_y, s_W, op);
    const Index y_bs1 = uniform_extent(*lay_y, s_H, op);
    const Index y_bs3 = uniform_extent(*lay_y, s_N, op);
    if(x_bs3 != y_bs3)
    {
        throw std::runtime_error(
            std::string("lower_to_tile ") + op +
            ": X and Y batch tile extents must match");
    }

    const auto& tiles_x = tile_lower::tiles_of(ctx.tile_map, X);
    const auto& tiles_c = tile_lower::tiles_of(ctx.tile_map, C);
    const auto& tiles_y = tile_lower::tiles_of(ctx.tile_map, Y);

    const Index Kx = C->shape()[0];
    const Index Ky = C->shape()[1];

    std::vector<Index> y_coord(g_WHCN);
    std::vector<Index> x_coord(g_WHCN);

    for(Index lin_y = 0; lin_y < lay_y->grid_volume(); ++lin_y)
    {
        lay_y->grid_coord_from_linear(lin_y, y_coord);
        TileGraph::TileNode* y_tile = tiles_y[static_cast<size_t>(lin_y)];
        const auto y_ts = lay_y->tile_shape_at(y_coord);

        Index y_lo_m = 0, y_hi_m = 0;
        Index y_lo_n = 0, y_hi_n = 0;
        lay_y->tile_axis_global_range(y_coord, s_W, y_lo_m, y_hi_m);
        lay_y->tile_axis_global_range(y_coord, s_H, y_lo_n, y_hi_n);
        const Index Y_start_m = y_lo_m;
        const Index Y_end_m = y_hi_m + 1;
        const Index Y_start_n = y_lo_n;
        const Index Y_end_n = y_hi_n + 1;

        Index X_start_m = stride[0] * Y_start_m - padding[0];
        Index X_end_m = stride[0] * (Y_end_m - 1) - padding[0]
            + dilation[0] * (Kx - 1) + 1;
        Index X_start_n = stride[1] * Y_start_n - padding[1];
        Index X_end_n = stride[1] * (Y_end_n - 1) - padding[1]
            + dilation[1] * (Ky - 1) + 1;

        const Index gx0 = lay_x->grid_shape()[s_W];
        const Index gx1 = lay_x->grid_shape()[s_H];
        Index X_start_tile_m = X_start_m / x_bs0;
        Index X_end_tile_m = (X_end_m - 1) / x_bs0 + 1;
        Index X_start_tile_n = X_start_n / x_bs1;
        Index X_end_tile_n = (X_end_n - 1) / x_bs1 + 1;

        if(X_end_tile_m <= 0 || X_start_tile_m >= gx0 || X_end_tile_n <= 0
            || X_start_tile_n >= gx1)
        {
            if(beta == 0.0)
            {
                tile::clear(y_tile);
            }
            else if(beta != 1.0)
            {
                tile::scale_inplace(beta, y_tile);
            }
            continue;
        }

        x_coord[s_C] = y_coord[s_C];
        x_coord[s_N] = y_coord[s_N];
        const Index start_m = std::max(X_start_tile_m, Index(0));
        const Index end_m = std::min(X_end_tile_m, gx0);
        const Index start_n = std::max(X_start_tile_n, Index(0));
        const Index end_n = std::min(X_end_tile_n, gx1);

        Scalar y_tile_beta = beta;
        for(Index X_i = start_m; X_i < end_m; ++X_i)
        {
            x_coord[s_W] = X_i;
            for(Index X_j = start_n; X_j < end_n; ++X_j)
            {
                x_coord[s_H] = X_j;
                const Index lin_x = lay_x->grid_linear(x_coord);
                const auto x_ts = lay_x->tile_shape_at(x_coord);
                const Index offset_m =
                    X_i * x_bs0 + padding[0] - stride[0] * Y_start_m;
                const Index offset_n =
                    X_j * x_bs1 + padding[1] - stride[1] * Y_start_n;
                tile::conv2d_inplace(
                    x_ts[s_W],
                    x_ts[s_H],
                    x_ts[s_C],
                    x_ts[s_N],
                    Kx,
                    Ky,
                    dilation[0],
                    dilation[1],
                    y_ts[s_C],
                    offset_m,
                    offset_n,
                    alpha,
                    tiles_x[static_cast<size_t>(lin_x)],
                    tiles_c[0],
                    y_ts[s_W],
                    y_ts[s_H],
                    stride[0],
                    stride[1],
                    y_tile_beta,
                    y_tile);
                y_tile_beta = 1.0;
            }
        }
    }
}

} // namespace nntile::tensor
