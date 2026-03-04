/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/scale_fiber.hh
 * TensorGraph scale_fiber operation: dst = alpha * src (fiber broadcast)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Scale fiber operation: dst = alpha * src (fiber broadcast)
struct TensorScaleFiberOp : TensorGraph::OpNode
{
    Scalar alpha;
    Index axis;
    Index batch_ndim;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorScaleFiberOp() = default;
    TensorScaleFiberOp(
        Scalar alpha_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_,
        Index batch_ndim_)
        : alpha(alpha_), axis(axis_), batch_ndim(batch_ndim_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SCALE_FIBER"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorScaleFiberOp>(*this);
    }
};

TensorGraph::TensorNode* scale_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    const std::vector<Index>& dst_shape,
    Index axis,
    Index batch_ndim);

void scale_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim);

} // namespace nntile::graph::tensor
