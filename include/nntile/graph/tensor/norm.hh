/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/norm.hh
 * TensorGraph norm operation: y = alpha * norm(x) + beta * y
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

//! Norm operation at tensor level
struct TensorNormOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* y = nullptr;

    TensorNormOp() = default;
    TensorNormOp(TensorGraph::TensorNode* x_, TensorGraph::TensorNode* y_,
                Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_)
    {
        inputs_ = {x, y};
        outputs_ = {y};
    }

    std::string op_name() const override { return "NORM"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorNormOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Euclidean norm: y = alpha * norm(x) + beta * y
void norm(TensorGraph::TensorNode* x, TensorGraph::TensorNode* y,
          Scalar alpha, Scalar beta);

} // namespace nntile::graph::tensor
