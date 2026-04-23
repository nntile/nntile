/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add.hh
 * TensorGraph add operation: z = alpha * x + beta * y
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Add operation at tensor level: z = alpha * x + beta * y
struct TensorAddOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* y = nullptr;
    TensorGraph::TensorNode* z = nullptr;

    TensorAddOp() = default;
    TensorAddOp(
        TensorGraph::TensorNode* x_,
        TensorGraph::TensorNode* y_,
        TensorGraph::TensorNode* z_,
        Scalar alpha_, Scalar beta_)
        : alpha(alpha_), beta(beta_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override { return "ADD"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Add operation: z = alpha * x + beta * y (creates output)
//! @param alpha Scaling factor for x
//! @param x First input tensor
//! @param beta Scaling factor for y
//! @param y Second input tensor
//! @param output_name Name for the output tensor
//! @return Pointer to the output tensor
TensorGraph::TensorNode* add(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y,
    const std::string& output_name);

//! Add operation: z = alpha * x + beta * y (uses existing output)
void add(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    Scalar beta,
    TensorGraph::TensorNode* y,
    TensorGraph::TensorNode* z);

} // namespace nntile::graph::tensor
