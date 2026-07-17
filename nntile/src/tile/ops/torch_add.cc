/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/tile/ops/torch_add.cc
 * TileGraph torch-native add.
 *
 * @version 1.1.0
 */

#include "nntile/tile/ops/torch_add.hh"

#include <stdexcept>

#include <nntile/core/torch_add.hh>
#include <nntile/runtime.hh>

namespace nntile::tile
{

void torch_add(
    TileGraph::TileNode *x,
    TileGraph::TileNode *y,
    TileGraph::TileNode *z,
    Scalar alpha
)
{
    if (x == nullptr || y == nullptr || z == nullptr)
    {
        throw std::invalid_argument(
            "tile torch_add: tiles must be non-null");
    }
    if (x->graph() != y->graph() || x->graph() != z->graph())
    {
        throw std::invalid_argument(
            "tile torch_add: tiles must share a graph");
    }
    if (x->dtype() != DataType::FP32 ||
        y->dtype() != DataType::FP32 ||
        z->dtype() != DataType::FP32)
    {
        throw std::invalid_argument(
            "tile torch_add: FP32 only");
    }
    if (x->shape() != y->shape() || x->shape() != z->shape())
    {
        throw std::invalid_argument(
            "tile torch_add: shape mismatch");
    }
    if (x == y || x == z || y == z)
    {
        throw std::invalid_argument(
            "tile torch_add: x, y, z must be distinct");
    }
    auto op = std::make_shared<TileTorchAddOp>(x, y, z, alpha);
    x->graph()->add_op(op);
}

void TileTorchAddOp::execute(Runtime &runtime) const
{
    auto &x_t = runtime.get_tile<fp32_t>(x);
    auto &y_t = runtime.get_tile<fp32_t>(y);
    auto &z_t = runtime.get_tile<fp32_t>(z);
    // Single-tile contiguous meta derived from tile shape (TensorNode /
    // TileNode full stride meta lands in a follow-up; see guide update).
    const core::TorchTileMeta meta =
        core::make_contiguous_torch_meta(x->shape());
    core::torch_add_out<fp32_t>(
        runtime.starpu_worker_hint(),
        x_t,
        meta,
        y_t,
        meta,
        z_t,
        meta,
        alpha);
}

} // namespace nntile::tile
