/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/conv2d_bwd_input_inplace.cc
 * TensorGraph conv2d_bwd_input_inplace operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/conv2d_bwd_input_inplace.hh"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/clear.hh"
#include "nntile/graph/tile/conv2d_bwd_input_inplace.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/scale_inplace.hh"
#include "nntile/tensor/conv2d_bwd_input_inplace.hh"

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



void conv2d_bwd_input_inplace(Scalar alpha,
                              TensorGraph::TensorNode* dY,
                              TensorGraph::TensorNode* kernel,
                              Scalar beta,
                              TensorGraph::TensorNode* dX,
                              std::array<Index, 2> padding,
                              std::array<Index, 2> stride,
                              std::array<Index, 2> dilation)
{
    if(dY == nullptr || kernel == nullptr || dX == nullptr)
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must be non-null");
    if(dY->graph() != kernel->graph() || kernel->graph() != dX->graph())
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must belong to same graph");
    if(dY->dtype() != kernel->dtype() || kernel->dtype() != dX->dtype())
        throw std::invalid_argument(
            "conv2d_bwd_input_inplace: tensors must have same dtype");
    auto op = std::make_shared<TensorConv2dBwdInputInplaceOp>(
        alpha, dY, kernel, beta, dX, padding, stride, dilation);
    dX->graph()->add_op(op);
}

void TensorConv2dBwdInputInplaceOp::lower_to_tile(
    const LoweringContext& ctx) const
{
    constexpr const char* op = "CONV2D_BWD_INPUT_INPLACE";
    const TensorAxisLayout* lay_dy = ctx.tiling.find(dY);
    const TensorAxisLayout* lay_k = ctx.tiling.find(kernel);
    const TensorAxisLayout* lay_dx = ctx.tiling.find(dX);
    if(lay_dy == nullptr || lay_k == nullptr || lay_dx == nullptr)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": missing tiling for dY/kernel/dX");
    }
    if(lay_k->grid_volume() != 1)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": kernel must be a single tile");
    }

    assert_full_in_channels(*lay_dy, op);
    assert_full_in_channels(*lay_dx, op);

    const Index dy_bs0 = uniform_extent(*lay_dy, 0, op);
    const Index dy_bs1 = uniform_extent(*lay_dy, 1, op);
    const Index dy_bs3 = uniform_extent(*lay_dy, 3, op);
    const Index dx_bs3 = uniform_extent(*lay_dx, 3, op);
    if(dy_bs3 != dx_bs3)
    {
        throw std::runtime_error(std::string("lower_to_tile ") + op +
            ": dY and dX batch tile extents must match");
    }

    const auto& tiles_dy = tile_lower::tiles_of(ctx.tile_map, dY);
    const auto& tiles_k = tile_lower::tiles_of(ctx.tile_map, kernel);
    const auto& tiles_dx = tile_lower::tiles_of(ctx.tile_map, dX);

    const Index Kx = kernel->shape()[0];
    const Index Ky = kernel->shape()[1];

    std::vector<Index> dx_coord(4);
    std::vector<Index> dy_coord(4);

    const Index gdy0 = lay_dy->grid_shape()[0];
    const Index gdy1 = lay_dy->grid_shape()[1];

    for(Index lin_dx = 0; lin_dx < lay_dx->grid_volume(); ++lin_dx)
    {
        lay_dx->grid_coord_from_linear(lin_dx, dx_coord);
        TileGraph::TileNode* dx_tile = tiles_dx[static_cast<size_t>(lin_dx)];
        const auto dx_ts = lay_dx->tile_shape_at(dx_coord);

        Index dx_lo_m = 0, dx_hi_m = 0;
        Index dx_lo_n = 0, dx_hi_n = 0;
        lay_dx->tile_axis_global_range(dx_coord, 0, dx_lo_m, dx_hi_m);
        lay_dx->tile_axis_global_range(dx_coord, 1, dx_lo_n, dx_hi_n);
        const Index dX_start_m = dx_lo_m;
        const Index dX_end_m = dx_hi_m + 1;
        const Index dX_start_n = dx_lo_n;
        const Index dX_end_n = dx_hi_n + 1;

        Index dY_start_m = (dX_start_m + padding[0] - dilation[0] * (Kx - 1)
                               + stride[0] - 1)
            / stride[0];
        Index dY_end_m =
            (dX_end_m - 1 + padding[0] + stride[0]) / stride[0];
        Index dY_start_n = (dX_start_n + padding[1] - dilation[1] * (Ky - 1)
                               + stride[1] - 1)
            / stride[1];
        Index dY_end_n =
            (dX_end_n - 1 + padding[1] + stride[1]) / stride[1];

        Index dY_start_tile_m = dY_start_m / dy_bs0;
        Index dY_end_tile_m = (dY_end_m - 1) / dy_bs0 + 1;
        Index dY_start_tile_n = dY_start_n / dy_bs1;
        Index dY_end_tile_n = (dY_end_n - 1) / dy_bs1 + 1;

        if(dY_end_tile_m <= 0 || dY_start_tile_m >= gdy0 || dY_end_tile_n <= 0
            || dY_start_tile_n >= gdy1)
        {
            if(beta == 0.0)
            {
                tile_graph::clear(dx_tile);
            }
            else if(beta != 1.0)
            {
                tile_graph::scale_inplace(beta, dx_tile);
            }
            continue;
        }

        dy_coord[2] = dx_coord[2];
        dy_coord[3] = dx_coord[3];
        const Index start_m = std::max(dY_start_tile_m, Index(0));
        const Index end_m = std::min(dY_end_tile_m, gdy0);
        const Index start_n = std::max(dY_start_tile_n, Index(0));
        const Index end_n = std::min(dY_end_tile_n, gdy1);

        Scalar dx_tile_beta = beta;
        for(Index dY_i = start_m; dY_i < end_m; ++dY_i)
        {
            dy_coord[0] = dY_i;
            for(Index dY_j = start_n; dY_j < end_n; ++dY_j)
            {
                dy_coord[1] = dY_j;
                const Index lin_dy = lay_dy->grid_linear(dy_coord);
                const auto dy_ts = lay_dy->tile_shape_at(dy_coord);
                const Index offset_m =
                    dX_start_m + padding[0] - stride[0] * dY_i * dy_bs0;
                const Index offset_n =
                    dX_start_n + padding[1] - stride[1] * dY_j * dy_bs1;
                tile_graph::conv2d_bwd_input_inplace(
                    dy_ts[0],
                    dy_ts[1],
                    stride[0],
                    stride[1],
                    dy_ts[2],
                    dy_ts[3],
                    Kx,
                    Ky,
                    dilation[0],
                    dilation[1],
                    dx_ts[2],
                    offset_m,
                    offset_n,
                    alpha,
                    tiles_dy[static_cast<size_t>(lin_dy)],
                    tiles_k[0],
                    dx_ts[0],
                    dx_ts[1],
                    dx_tile_beta,
                    dx_tile);
                dx_tile_beta = 1.0;
            }
        }
    }
}

} // namespace nntile::graph::tensor
