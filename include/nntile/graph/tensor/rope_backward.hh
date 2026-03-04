/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/rope_backward.hh
 * TensorGraph rope_backward operation: (sin, cos, dy, dx)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! RoPE backward operation: dx = rope_backward(sin, cos, dy)
struct TensorRopeBackwardOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* sin = nullptr;
    TensorGraph::TensorNode* cos = nullptr;
    TensorGraph::TensorNode* dy = nullptr;
    TensorGraph::TensorNode* dx = nullptr;

    TensorRopeBackwardOp() = default;
    TensorRopeBackwardOp(
        TensorGraph::TensorNode* sin_,
        TensorGraph::TensorNode* cos_,
        TensorGraph::TensorNode* dy_,
        TensorGraph::TensorNode* dx_)
        : sin(sin_), cos(cos_), dy(dy_), dx(dx_)
    {
        inputs_ = {sin, cos, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "ROPE_BACKWARD"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorRopeBackwardOp>(*this);
    }
};

TensorGraph::TensorNode* rope_backward(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* dy,
    const std::string& output_name);

void rope_backward(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx);

} // namespace nntile::graph::tensor
