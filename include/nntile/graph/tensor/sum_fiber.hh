/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/sum_fiber.hh
 * TensorGraph sum_fiber operation: y = alpha * sum_fiber(x) + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Sum fiber operation at tensor level: y = alpha * sum_fiber(x) + beta * y
struct TensorSumFiberOp : TensorGraph::OpNode
{
    Index axis;
    Index batch_ndim;
    int redux;
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* y = nullptr;

    TensorSumFiberOp() = default;
    TensorSumFiberOp(
        TensorGraph::TensorNode* x_,
        TensorGraph::TensorNode* y_,
        Index axis_, Index batch_ndim_,
        int redux_, Scalar alpha_, Scalar beta_)
        : axis(axis_), batch_ndim(batch_ndim_)
        , redux(redux_), alpha(alpha_), beta(beta_)
        , x(x_), y(y_)
    {
        inputs_ = {x, y};
        outputs_ = {y};
    }

    std::string op_name() const override { return "SUM_FIBER"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSumFiberOp>(*this);
    }
};

//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to sum (default: 0)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void sum_fiber(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    Index axis,
    Index batch_ndim,
    int redux,
    Scalar alpha,
    Scalar beta);

} // namespace nntile::graph
