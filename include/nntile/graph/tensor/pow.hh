/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/pow.hh
 * TensorGraph pow operation: (alpha, exp, A) in-place
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

//! Pow in-place: A = alpha * A^exp
struct TensorPowOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar exp;
    TensorGraph::TensorNode* A = nullptr;

    TensorPowOp() = default;
    TensorPowOp(
        Scalar alpha_,
        Scalar exp_,
        TensorGraph::TensorNode* A_)
        : alpha(alpha_), exp(exp_)
        , A(A_)
    {
        inputs_ = {A};
        outputs_ = {A};
    }

    std::string op_name() const override { return "POW"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorPowOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

void pow(
    Scalar alpha,
    Scalar exp,
    TensorGraph::TensorNode* A);

} // namespace nntile::graph::tensor
