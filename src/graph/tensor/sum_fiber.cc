/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sum_fiber.cc
 * TensorGraph sum_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sum_fiber.hh"

#include <stdexcept>
#include <utility>
#include <vector>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/sum_fiber.hh"

#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/sum_fiber.hh"

namespace nntile::graph::tensor
{

namespace
{

std::vector<Index> sum_fiber_output_shape(
    const std::vector<Index>& x_shape,
    Index axis,
    Index batch_ndim)
{
    Index ndim = x_shape.size();
    std::vector<Index> out_shape;
    out_shape.reserve(batch_ndim + 1);
    out_shape.push_back(x_shape[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        out_shape.push_back(x_shape[ndim - batch_ndim + i]);
    }
    return out_shape;
}

} // namespace

TensorGraph::TensorNode* sum_fiber(
    TensorGraph::TensorNode* x,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr)
    {
        throw std::invalid_argument(
            "sum_fiber: input tensor must be non-null");
    }

    std::vector<Index> output_shape =
        sum_fiber_output_shape(x->shape(), axis, batch_ndim);
    TensorGraph::TensorNode* output = x->graph()->data(
        std::move(output_shape),
        output_name,
        x->dtype());

    // Merge output fiber axes with x axes
    merge_axis(output->mutable_axes()[0],
               x->mutable_axes()[axis]);
    for(Index i = 0; i < batch_ndim; ++i)
    {
        merge_axis(output->mutable_axes()[1 + i],
                   x->mutable_axes()[x->ndim() - batch_ndim + i]);
    }

    auto op = std::make_shared<TensorSumFiberOp>(
        x, output, axis, batch_ndim, redux, alpha, beta);
    x->graph()->add_op(op);

    return output;
}

void sum_fiber(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must be non-null");
    }
    if(x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must belong to the same graph");
    }
    if(x->dtype() != y->dtype())
    {
        throw std::invalid_argument(
            "sum_fiber: input tensors must have the same dtype");
    }
    if(x == y)
    {
        throw std::invalid_argument(
            "sum_fiber: x and y must be distinct tensors");
    }
    validate_fiber_shape_and_merge(y, x, axis, batch_ndim, "sum_fiber");

    auto op = std::make_shared<TensorSumFiberOp>(
        x, y, axis, batch_ndim, redux, alpha, beta);

    x->graph()->add_op(op);
}

void TensorSumFiberOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::sum_fiber_async (src/tensor/sum_fiber.cc).
    const TensorAxisLayout* lay_x = ctx.tiling.find(x);
    const TensorAxisLayout* lay_y = ctx.tiling.find(y);
    if(lay_x == nullptr || lay_y == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SUM_FIBER: missing tiling for x and/or y");
    }

    const auto& tiles_x = tile_lower::tiles_of(ctx.tile_map, x);
    const auto& tiles_y = tile_lower::tiles_of(ctx.tile_map, y);

    std::vector<Index> x_coord;
    std::vector<Index> y_coord(static_cast<size_t>(y->ndim()));

    const Index fiber_prefix = x->ndim() - batch_ndim;

    for(Index lin_x = 0; lin_x < lay_x->grid_volume(); ++lin_x)
    {
        lay_x->grid_coord_from_linear(lin_x, x_coord);
        TileGraph::TileNode* x_tile = tiles_x[static_cast<size_t>(lin_x)];

        y_coord[0] = x_coord[static_cast<size_t>(axis)];
        for(Index j = 0; j < batch_ndim; ++j)
        {
            y_coord[static_cast<size_t>(j + 1)] =
                x_coord[static_cast<size_t>(x->ndim() - batch_ndim + j)];
        }
        const Index lin_y = lay_y->grid_linear(y_coord);
        TileGraph::TileNode* y_tile = tiles_y[static_cast<size_t>(lin_y)];

        bool init_first = true;
        for(Index j = 0; j < fiber_prefix; ++j)
        {
            if(j != axis && x_coord[static_cast<size_t>(j)] != 0)
            {
                init_first = false;
                break;
            }
        }

        if(init_first)
        {
            tile_graph::sum_fiber(
                alpha, x_tile, beta, y_tile, axis, batch_ndim, redux);
        }
        else
        {
            tile_graph::sum_fiber(
                alpha, x_tile, Scalar(1.0), y_tile, axis, batch_ndim, redux);
        }
    }
}

} // namespace nntile::graph::tensor
