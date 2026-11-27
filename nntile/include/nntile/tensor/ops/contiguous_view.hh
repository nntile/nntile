/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/tensor/ops/contiguous_view.hh
 * TensorGraph contiguous_view: dst is a storage alias of src.
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

struct TensorContiguousViewOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode *src = nullptr;
    TensorGraph::TensorNode *dst = nullptr;

    TensorContiguousViewOp() = default;
    TensorContiguousViewOp(
        TensorGraph::TensorNode *src_,
        TensorGraph::TensorNode *dst_) :
        src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "CONTIGUOUS_VIEW"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorContiguousViewOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void contiguous_view(
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst);

} // namespace nntile::tensor
