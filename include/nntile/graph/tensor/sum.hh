/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/sum.hh
 * TensorGraph sum operation: dst = alpha * sum(src) + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Sum operation at tensor level: dst = alpha * sum(src) + beta * dst
struct TensorSumOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorSumOp() = default;
    TensorSumOp(
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SUM"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSumOp>(*this);
    }
};

//! Sum all elements: dst = alpha * sum(src) + beta * dst
void sum(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Scalar beta);

} // namespace nntile::graph::tensor
