/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/multiply.hh
 * TensorGraph multiply operation: z = x * y
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Multiply operation at tensor level
struct TensorMultiplyOp : TensorGraphOpNode
{
    Scalar alpha = 1.0;
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* y = nullptr;
    TensorGraph::TensorNode* z = nullptr;

    TensorMultiplyOp() = default;
    TensorMultiplyOp(TensorGraph::TensorNode* x_, TensorGraph::TensorNode* y_,
                    TensorGraph::TensorNode* z_, Scalar alpha_ = 1.0)
        : alpha(alpha_), x(x_), y(y_), z(z_)
    {
        inputs_ = {x, y};
        outputs_ = {z};
    }

    std::string op_name() const override { return "MULTIPLY"; }

    void execute(ExecutionContext<TensorGraph::TensorNode>& ctx) const override;

    std::shared_ptr<TensorGraphOpNode> clone() const override
    {
        return std::make_shared<TensorMultiplyOp>(*this);
    }
};

//! Multiply: z = x * y
//! @return Pointer to the output tensor
TensorGraph::TensorNode* multiply(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y,
    const std::string& output_name);

} // namespace nntile::graph
