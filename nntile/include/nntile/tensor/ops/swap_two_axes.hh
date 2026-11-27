/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file include/nntile/tensor/ops/swap_two_axes.hh
 * TensorGraph swap_two_axes operation.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

struct TensorSwapTwoAxesOp : TensorGraph::OpNode
{
    Index dim0 = 0;
    Index dim1 = 0;
    TensorGraph::TensorNode *src = nullptr;
    TensorGraph::TensorNode *dst = nullptr;

    TensorSwapTwoAxesOp() = default;
    TensorSwapTwoAxesOp(
        TensorGraph::TensorNode *src_,
        TensorGraph::TensorNode *dst_,
        Index dim0_,
        Index dim1_) :
        dim0(dim0_),
        dim1(dim1_),
        src(src_),
        dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SWAP_TWO_AXES"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSwapTwoAxesOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void swap_two_axes(
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    Index dim0,
    Index dim1);

} // namespace nntile::tensor
