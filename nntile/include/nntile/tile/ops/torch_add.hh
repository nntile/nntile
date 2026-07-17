/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tile/ops/torch_add.hh
 * TileGraph torch-native add (single-tile).
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/tile/ops/torch_add.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/tile/graph.hh>

namespace nntile::tile
{

//! Single-tile torch add: z = x + alpha * y
struct TileTorchAddOp : TileGraph::OpNode
{
    Scalar alpha = 1.0;
    TileGraph::TileNode *x = nullptr;
    TileGraph::TileNode *y = nullptr;
    TileGraph::TileNode *z = nullptr;

    TileTorchAddOp() = default;
    TileTorchAddOp(
        TileGraph::TileNode *x_,
        TileGraph::TileNode *y_,
        TileGraph::TileNode *z_,
        Scalar alpha_) :
        alpha(alpha_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override
    {
        return "TILE_TORCH_ADD";
    }

    void execute(Runtime &runtime) const override;

    std::shared_ptr<TileGraph::OpNode> clone() const override
    {
        return std::make_shared<TileTorchAddOp>(*this);
    }
};

void torch_add(
    TileGraph::TileNode *x,
    TileGraph::TileNode *y,
    TileGraph::TileNode *z,
    Scalar alpha
);

} // namespace nntile::tile
