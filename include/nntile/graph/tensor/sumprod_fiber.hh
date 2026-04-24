/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/sumprod_fiber.hh
 * TensorGraph sumprod_fiber operation: dst = alpha * sumprod_fiber(src1, src2) + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Sumprod fiber operation: dst = alpha * sumprod_fiber(src1, src2) + beta * dst
struct TensorSumprodFiberOp : TensorGraph::OpNode
{
    Index axis;
    int redux;
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorSumprodFiberOp() = default;
    TensorSumprodFiberOp(
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_,
        Scalar alpha_, Scalar beta_)
        : axis(axis_), redux(redux_), alpha(alpha_), beta(beta_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SUMPROD_FIBER"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSumprodFiberOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Sumprod over fibers: dst = alpha * sumprod_fiber(src1, src2) + beta * dst
void sumprod_fiber(
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta);

} // namespace nntile::graph::tensor
