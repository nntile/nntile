/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/add_fiber.cc
 * TensorGraph add_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/add_fiber.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/add_fiber.hh"

#include "nntile/graph/tile/add_fiber.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* add_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must be non-null");
    }
    if(fiber->graph() != tensor->graph())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must belong to the same graph");
    }
    if(fiber->dtype() != tensor->dtype())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must have the same dtype");
    }

    validate_fiber_shape_and_merge(fiber, tensor, axis, batch_ndim, "add_fiber");

    // Output shape matches tensor (fiber is broadcast)
    std::vector<Index> output_shape = tensor->shape();
    TensorGraph::TensorNode* output = tensor->graph()->data(
        std::move(output_shape),
        output_name,
        tensor->dtype());
    output->set_axes(tensor->axes());

    auto op = std::make_shared<TensorAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    fiber->graph()->add_op(op);

    return output;
}

void add_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    TensorGraph::TensorNode* output,
    Index axis,
    Index batch_ndim)
{
    if(fiber == nullptr || tensor == nullptr || output == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must be non-null");
    }
    if(fiber->graph() != tensor->graph() || fiber->graph() != output->graph())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must belong to the same graph");
    }
    if(fiber->dtype() != tensor->dtype() || fiber->dtype() != output->dtype())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must have the same dtype");
    }
    if(fiber == tensor || fiber == output || tensor == output)
    {
        throw std::invalid_argument(
            "add_fiber: fiber, tensor, and output must be distinct tensors");
    }
    validate_fiber_shape_and_merge(fiber, tensor, axis, batch_ndim, "add_fiber");
    validate_same_shape_and_merge(tensor, output, "add_fiber");

    auto op = std::make_shared<TensorAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    fiber->graph()->add_op(op);
}

void TensorAddFiberOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::add_fiber_async (src/tensor/add_fiber.cc).
    const TensorAxisLayout* lay_d = ctx.tiling.find(output);
    const TensorAxisLayout* lay_f = ctx.tiling.find(fiber);
    if(lay_d == nullptr || lay_f == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ADD_FIBER: missing tiling for output and/or fiber");
    }

    tile_lower::assert_same_elementwise_layout(tensor, output, "ADD_FIBER tensor/output");

    const auto& tiles_f = tile_lower::tiles_of(ctx.tile_map, fiber);
    const auto& tiles_t = tile_lower::tiles_of(ctx.tile_map, tensor);
    const auto& tiles_o = tile_lower::tiles_of(ctx.tile_map, output);

    std::vector<Index> dst_coord;
    std::vector<Index> fiber_coord(static_cast<size_t>(fiber->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        const Index j = dst_coord[static_cast<size_t>(axis)];
        fiber_coord[0] = j;
        for(Index b = 0; b < batch_ndim; ++b)
        {
            fiber_coord[static_cast<size_t>(b + 1)] =
                dst_coord[static_cast<size_t>(output->ndim() - batch_ndim + b)];
        }
        const Index lin_f = lay_f->grid_linear(fiber_coord);
        tile_graph::add_fiber(
            alpha,
            tiles_f[static_cast<size_t>(lin_f)],
            beta,
            tiles_t[static_cast<size_t>(lin_d)],
            tiles_o[static_cast<size_t>(lin_d)],
            axis,
            batch_ndim);
    }
}

} // namespace nntile::graph::tensor
