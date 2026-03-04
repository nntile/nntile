/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/hypot_scalar_inverse.hh
 * TensorGraph hypot_scalar_inverse operation: (eps, alpha, dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Hypot scalar inverse: dst = 1/hypot(alpha*dst, eps)
struct TensorHypotScalarInverseOp : TensorGraph::OpNode
{
    Scalar eps;
    Scalar alpha;
    TensorGraph::TensorNode* dst = nullptr;

    TensorHypotScalarInverseOp() = default;
    TensorHypotScalarInverseOp(
        Scalar eps_,
        Scalar alpha_,
        TensorGraph::TensorNode* dst_)
        : eps(eps_), alpha(alpha_)
        , dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "HYPOT_SCALAR_INVERSE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorHypotScalarInverseOp>(*this);
    }
};

void hypot_scalar_inverse(
    Scalar eps,
    Scalar alpha,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
