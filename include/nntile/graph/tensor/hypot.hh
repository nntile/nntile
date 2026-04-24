/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/hypot.hh
 * TensorGraph hypot operation: (alpha, src1, beta, src2, dst)
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

//! Hypot operation: dst = hypot(alpha*src1, beta*src2)
struct TensorHypotOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorHypotOp() = default;
    TensorHypotOp(
        Scalar alpha_, Scalar beta_,
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_)
        : alpha(alpha_), beta(beta_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "HYPOT"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorHypotOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

TensorGraph::TensorNode* hypot(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name);

void hypot(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
