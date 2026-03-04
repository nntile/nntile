/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_inplace.hh
 * TensorGraph add_inplace operation: y = alpha * x + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Add in-place operation at tensor level: y = alpha * x + beta * y
struct TensorAddInplaceOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* y = nullptr;

    TensorAddInplaceOp() = default;
    TensorAddInplaceOp(
        TensorGraph::TensorNode* x_,
        TensorGraph::TensorNode* y_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_)
    {
        inputs_ = {x, y};
        outputs_ = {y};
    }

    std::string op_name() const override { return "ADD_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddInplaceOp>(*this);
    }
};

//! Add in-place: y = alpha * x + beta * y
void add_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y);

} // namespace nntile::graph::tensor
