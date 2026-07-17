/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/tensor/ops/torch_add.cc
 * TensorGraph torch-native add (enforced single tile).
 *
 * @version 1.1.0
 */

#include "nntile/tensor/ops/torch_add.hh"

#include <stdexcept>
#include <utility>

#include <nntile/base_types.hh>
#include <nntile/tensor/tile_lowering_helpers.hh>
#include <nntile/tile/ops/torch_add.hh>

namespace nntile::tensor
{

TensorGraph::TensorNode *torch_add(
    TensorGraph::TensorNode *x,
    TensorGraph::TensorNode *y,
    Scalar alpha
)
{
    if (x == nullptr || y == nullptr)
    {
        throw std::invalid_argument(
            "torch_add: inputs must be non-null");
    }
    if (x->graph() != y->graph())
    {
        throw std::invalid_argument(
            "torch_add: inputs must share a graph");
    }
    if (x == y)
    {
        throw std::invalid_argument(
            "torch_add: x and y must be distinct");
    }
    if (x->dtype() != DataType::FP32 || y->dtype() != DataType::FP32)
    {
        throw std::invalid_argument("torch_add: FP32 only");
    }
    validate_same_shape_and_merge(x, y, "torch_add");

    TensorGraph::TensorNode *output =
        x->graph()->emplace_data(x->shape(), x->dtype());
    output->set_axes(x->axes());

    auto op = std::make_shared<TensorTorchAddOp>(
        x,
        y,
        output,
        alpha);
    x->graph()->add_op(op);
    return output;
}

void torch_add(
    TensorGraph::TensorNode *x,
    TensorGraph::TensorNode *y,
    TensorGraph::TensorNode *z,
    Scalar alpha
)
{
    if (x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument(
            "torch_add: tensors must be non-null");
    }
    if (x == y || x == z || y == z)
    {
        throw std::invalid_argument(
            "torch_add: x, y, z must be distinct");
    }
    if (x->graph() != y->graph() || x->graph() != z->graph())
    {
        throw std::invalid_argument(
            "torch_add: tensors must share a graph");
    }
    if (x->dtype() != DataType::FP32 ||
        y->dtype() != DataType::FP32 ||
        z->dtype() != DataType::FP32)
    {
        throw std::invalid_argument("torch_add: FP32 only");
    }
    validate_same_shape_and_merge(x, y, "torch_add");
    validate_same_shape_and_merge(x, z, "torch_add");

    auto op = std::make_shared<TensorTorchAddOp>(x, y, z, alpha);
    x->graph()->add_op(op);
}

void TensorTorchAddOp::lower_to_tile(const LoweringContext &ctx) const
{
    const auto &m = ctx.tile_map;
    const auto &vx = tile_lower::tiles_of(m, x);
    const auto &vy = tile_lower::tiles_of(m, y);
    const auto &vz = tile_lower::tiles_of(m, z);
    if (vx.size() != 1 || vy.size() != 1 || vz.size() != 1)
    {
        throw std::runtime_error(
            "lower_to_tile TORCH_ADD: requires exactly one tile "
            "per operand (untiled tensors only)");
    }
    tile_lower::assert_same_elementwise_layout(x, y, "TORCH_ADD x/y");
    tile_lower::assert_same_elementwise_layout(x, z, "TORCH_ADD x/z");
    tile::torch_add(vx[0], vy[0], vz[0], alpha);
}

} // namespace nntile::tensor
