/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tensor/ops/torch_add.hh
 * TensorGraph torch-native add (untiled / single-tile only).
 *
 * @version 1.1.0
 */

#pragma once

#include <nntile/defs.h>

#ifndef NNTILE_TORCH_NATIVE_OPS
#error "nntile/tensor/ops/torch_add.hh requires NNTILE_TORCH_NATIVE_OPS"
#endif

#include <string>
#include <vector>

#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! Torch-native add at TensorGraph: z = x + alpha * y
struct TensorTorchAddOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    TensorGraph::TensorNode *x = nullptr;
    TensorGraph::TensorNode *y = nullptr;
    TensorGraph::TensorNode *z = nullptr;

    TensorTorchAddOp() = default;
    TensorTorchAddOp(
        TensorGraph::TensorNode *x_,
        TensorGraph::TensorNode *y_,
        TensorGraph::TensorNode *z_,
        Scalar alpha_) :
        alpha(alpha_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override
    {
        return "TORCH_ADD";
    }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTorchAddOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

//! Record z = x + alpha * y (creates output node).
TensorGraph::TensorNode *torch_add(
    TensorGraph::TensorNode *x,
    TensorGraph::TensorNode *y,
    Scalar alpha
);

//! Record z = x + alpha * y into existing output.
void torch_add(
    TensorGraph::TensorNode *x,
    TensorGraph::TensorNode *y,
    TensorGraph::TensorNode *z,
    Scalar alpha
);

} // namespace nntile::tensor
