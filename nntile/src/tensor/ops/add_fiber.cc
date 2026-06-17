#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/add_fiber.cc
 * TensorGraph add_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/add_fiber.hh"

#include "nntile/base_types.hh"
#include "nntile/tensor/shape_layout.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/add_fiber.hh"
#include "nntile/tensor/ops/add_fiber.hh"

#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

TensorGraph::TensorNode *add_fiber(Scalar alpha,
    TensorGraph::TensorNode *fiber,
    Scalar beta,
    TensorGraph::TensorNode *tensor,
    Index axis,
    Index batch_ndim)
{
    if (fiber == nullptr || tensor == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must be non-null");
    }
    if (fiber->graph() != tensor->graph())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must belong to the same graph");
    }
    if (fiber->dtype() != tensor->dtype())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must have the same dtype");
    }

    validate_fiber_shape_and_merge(
        fiber, tensor, axis, batch_ndim, "add_fiber");

    // Output shape matches tensor (fiber is broadcast)
    std::vector<Index> output_shape = tensor->shape();
    TensorGraph::TensorNode *output =
        tensor->graph()->data(std::move(output_shape), tensor->dtype());
    output->set_axes(tensor->axes());

    auto op = std::make_shared<TensorAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    fiber->graph()->add_op(op);

    return output;
}

void add_fiber(Scalar alpha,
    TensorGraph::TensorNode *fiber,
    Scalar beta,
    TensorGraph::TensorNode *tensor,
    TensorGraph::TensorNode *output,
    Index axis,
    Index batch_ndim)
{
    if (fiber == nullptr || tensor == nullptr || output == nullptr)
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must be non-null");
    }
    if (fiber->graph() != tensor->graph() || fiber->graph() != output->graph())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must belong to the same graph");
    }
    if (fiber->dtype() != tensor->dtype() || fiber->dtype() != output->dtype())
    {
        throw std::invalid_argument(
            "add_fiber: input tensors must have the same dtype");
    }
    if (fiber == tensor || fiber == output || tensor == output)
    {
        throw std::invalid_argument(
            "add_fiber: fiber, tensor, and output must be distinct tensors");
    }

    validate_fiber_shape_and_merge(
        fiber, tensor, axis, batch_ndim, "add_fiber");
    validate_same_shape_and_merge(tensor, output, "add_fiber");

    auto op = std::make_shared<TensorAddFiberOp>(
        fiber, tensor, output, alpha, beta, axis, batch_ndim);
    fiber->graph()->add_op(op);
}

void TensorAddFiberOp::lower_to_tile(const LoweringContext &ctx) const
{
    // Match nntile::tensor::add_fiber_async (src/tensor/add_fiber.cc).
    const TensorAxisLayout *lay_d = ctx.tiling.find(output);
    const TensorAxisLayout *lay_f = ctx.tiling.find(fiber);
    if (lay_d == nullptr || lay_f == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ADD_FIBER: missing tiling for output and/or fiber");
    }

    tile_lower::assert_same_elementwise_layout(
        tensor, output, "ADD_FIBER tensor/output");

    const auto &tiles_f = tile_lower::tiles_of(ctx.tile_map, fiber);
    const auto &tiles_t = tile_lower::tiles_of(ctx.tile_map, tensor);
    const auto &tiles_o = tile_lower::tiles_of(ctx.tile_map, output);

    const Index out_nd = output->ndim();
    const Index fiber_nd = fiber->ndim();

    std::vector<Index> dst_coord;
    std::vector<Index> fiber_coord(static_cast<size_t>(fiber_nd));

    for (Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        fiber_layout_coord_from_tensor(
            dst_coord, axis, batch_ndim, fiber_nd, out_nd, fiber_coord);
        const Index lin_f = lay_f->grid_linear(fiber_coord);
        tile::add_fiber(alpha,
            tiles_f[static_cast<size_t>(lin_f)],
            beta,
            tiles_t[static_cast<size_t>(lin_d)],
            tiles_o[static_cast<size_t>(lin_d)],
            axis,
            batch_ndim);
    }
}

} // namespace nntile::tensor
