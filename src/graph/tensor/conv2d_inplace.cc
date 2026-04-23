/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/conv2d_inplace.cc
 * TensorGraph conv2d_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/conv2d_inplace.hh"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/clear.hh"
#include "nntile/graph/tile/conv2d_inplace.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/scale_inplace.hh"
#include "nntile/tensor/conv2d_inplace.hh"

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
            ": channel axis must be a single tile (full C_in)");
    }
    const Index ext = uniform_extent(lay, 2, op);
    if(ext != lay.tensor_shape()[2])
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

    const Index x_bs0 = uniform_extent(*lay_x, 0, op);
    const Index x_bs1 = uniform_extent(*lay_x, 1, op);
    const Index x_bs3 = uniform_extent(*lay_x, 3, op);
    const Index y_bs0 = uniform_extent(*lay_y, 0, op);
    const Index y_bs1 = uniform_extent(*lay_y, 1, op);
    const Index y_bs3 = uniform_extent(*lay_y, 3, op);
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

    std::vector<Index> y_coord(4);
    std::vector<Index> x_coord(4);

    for(Index lin_y = 0; lin_y < lay_y->grid_volume(); ++lin_y)
    {
        lay_y->grid_coord_from_linear(lin_y, y_coord);
        TileGraph::TileNode* y_tile = tiles_y[static_cast<size_t>(lin_y)];
        const auto y_ts = lay_y->tile_shape_at(y_coord);

        Index y_lo_m = 0, y_hi_m = 0;
        Index y_lo_n = 0, y_hi_n = 0;
        lay_y->tile_axis_global_range(y_coord, 0, y_lo_m, y_hi_m);
        lay_y->tile_axis_global_range(y_coord, 1, y_lo_n, y_hi_n);
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

        const Index gx0 = lay_x->grid_shape()[0];
        const Index gx1 = lay_x->grid_shape()[1];
        Index X_start_tile_m = X_start_m / x_bs0;
        Index X_end_tile_m = (X_end_m - 1) / x_bs0 + 1;
        Index X_start_tile_n = X_start_n / x_bs1;
        Index X_end_tile_n = (X_end_n - 1) / x_bs1 + 1;

        if(X_end_tile_m <= 0 || X_start_tile_m >= gx0 || X_end_tile_n <= 0
            || X_start_tile_n >= gx1)
        {
            if(beta == 0.0)
            {
                tile_graph::clear(y_tile);
            }
            else if(beta != 1.0)
            {
                tile_graph::scale_inplace(beta, y_tile);
            }
            continue;
        }

        x_coord[2] = y_coord[2];
        x_coord[3] = y_coord[3];
        const Index start_m = std::max(X_start_tile_m, Index(0));
        const Index end_m = std::min(X_end_tile_m, gx0);
        const Index start_n = std::max(X_start_tile_n, Index(0));
        const Index end_n = std::min(X_end_tile_n, gx1);

        Scalar y_tile_beta = beta;
        for(Index X_i = start_m; X_i < end_m; ++X_i)
        {
            x_coord[0] = X_i;
            for(Index X_j = start_n; X_j < end_n; ++X_j)
            {
                x_coord[1] = X_j;
                const Index lin_x = lay_x->grid_linear(x_coord);
                const auto x_ts = lay_x->tile_shape_at(x_coord);
                const Index offset_m =
                    X_i * x_bs0 + padding[0] - stride[0] * Y_start_m;
                const Index offset_n =
                    X_j * x_bs1 + padding[1] - stride[1] * Y_start_n;
                tile_graph::conv2d_inplace(
                    x_ts[0],
                    x_ts[1],
                    x_ts[2],
                    x_ts[3],
                    Kx,
                    Ky,
                    dilation[0],
                    dilation[1],
                    y_ts[2],
                    offset_m,
                    offset_n,
                    alpha,
                    tiles_x[static_cast<size_t>(lin_x)],
                    tiles_c[0],
                    y_ts[0],
                    y_ts[1],
                    stride[0],
                    stride[1],
                    y_tile_beta,
                    y_tile);
                y_tile_beta = 1.0;
            }
        }
    }
}

} // namespace nntile::graph::tensor
