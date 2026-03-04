/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_fiber.hh
 * TensorGraph add_fiber operation: output = alpha * fiber + beta * tensor
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Add fiber operation at tensor level: output = alpha * fiber + beta * tensor
struct TensorAddFiberOp : TensorGraph::OpNode
{
    Index axis;
    Index batch_ndim;
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* fiber = nullptr;
    TensorGraph::TensorNode* tensor = nullptr;
    TensorGraph::TensorNode* output = nullptr;

    TensorAddFiberOp() = default;
    TensorAddFiberOp(
        TensorGraph::TensorNode* fiber_,
        TensorGraph::TensorNode* tensor_,
        TensorGraph::TensorNode* output_,
        Scalar alpha_, Scalar beta_,
        Index axis_, Index batch_ndim_)
        : axis(axis_), batch_ndim(batch_ndim_)
        , alpha(alpha_), beta(beta_)
        , fiber(fiber_), tensor(tensor_), output(output_)
    {
        inputs_ = {fiber, tensor};
        outputs_ = {output};
    }

    std::string op_name() const override { return "ADD_FIBER"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddFiberOp>(*this);
    }
};

//! Add along fibers: output = alpha * fiber + beta * tensor (creates output)
TensorGraph::TensorNode* add_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    const std::string& output_name,
    Index axis,
    Index batch_ndim);

//! Add along fibers: output = alpha * fiber + beta * tensor (uses existing output)
void add_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* fiber,
    Scalar beta,
    TensorGraph::TensorNode* tensor,
    TensorGraph::TensorNode* output,
    Index axis,
    Index batch_ndim);

} // namespace nntile::graph::tensor
