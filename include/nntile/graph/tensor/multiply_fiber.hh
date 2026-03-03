/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/multiply_fiber.hh
 * TensorGraph multiply_fiber operation: dst = alpha * src1 * src2 (fiber)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Multiply fiber operation: dst = alpha * src1 * src2
struct TensorMultiplyFiberOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    Index axis = 0;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorMultiplyFiberOp() = default;
    TensorMultiplyFiberOp(
        Scalar alpha_,
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_,
        Index axis_)
        : alpha(alpha_), axis(axis_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "MULTIPLY_FIBER"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMultiplyFiberOp>(*this);
    }
};

TensorGraph::TensorNode* multiply_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis);

void multiply_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis);

} // namespace nntile::graph
