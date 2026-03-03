/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_fiber_inplace.hh
 * TensorGraph add_fiber_inplace: tensor = alpha * fiber + beta * tensor
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Add fiber in-place at tensor level: tensor = alpha * fiber + beta * tensor
struct TensorAddFiberInplaceOp : TensorGraph::OpNode
{
    Index axis = 0;
    Index batch_ndim = 0;
    Scalar alpha = 1.0;
    Scalar beta = 1.0;
    TensorGraph::TensorNode* fiber = nullptr;
    TensorGraph::TensorNode* tensor = nullptr;

    TensorAddFiberInplaceOp() = default;
    TensorAddFiberInplaceOp(
        TensorGraph::TensorNode* fiber_,
        TensorGraph::TensorNode* tensor_,
        Scalar alpha_, Scalar beta_,
        Index axis_, Index batch_ndim_)
        : axis(axis_), batch_ndim(batch_ndim_)
        , alpha(alpha_), beta(beta_)
        , fiber(fiber_), tensor(tensor_)
    {
        inputs_ = {fiber, tensor};
        outputs_ = {tensor};
    }

    std::string op_name() const override { return "ADD_FIBER_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddFiberInplaceOp>(*this);
    }
};

//! Add along fibers in-place: tensor = alpha * fiber + beta * tensor
void add_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    Index axis,
    Index batch_ndim = 0);

} // namespace nntile::graph
