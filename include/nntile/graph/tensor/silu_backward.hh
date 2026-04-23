/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/silu_backward.hh
 * TensorGraph silu_backward operation: dx = silu_backward(x, dy)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! SiLU backward operation: dx = silu_backward(x, dy)
struct TensorSiluBackwardOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* dy = nullptr;
    TensorGraph::TensorNode* dx = nullptr;

    TensorSiluBackwardOp() = default;
    TensorSiluBackwardOp(
        TensorGraph::TensorNode* x_,
        TensorGraph::TensorNode* dy_,
        TensorGraph::TensorNode* dx_)
        : x(x_), dy(dy_), dx(dx_)
    {
        inputs_ = {x, dy, dx};
        outputs_ = {dx};
    }

    std::string op_name() const override { return "SILU_BACKWARD"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSiluBackwardOp>(*this);
    }
    void lower_to_tile(const LoweringContext& ctx) const override;

};

TensorGraph::TensorNode* silu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    const std::string& output_name);

void silu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx);

} // namespace nntile::graph::tensor
