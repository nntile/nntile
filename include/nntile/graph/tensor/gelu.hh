/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/gelu.hh
 * TensorGraph GeLU operation: y = gelu(x)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! GeLU operation at tensor level: y = gelu(x)
struct TensorGeluOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* x = nullptr;
    TensorGraph::TensorNode* y = nullptr;

    TensorGeluOp() = default;
    TensorGeluOp(TensorGraph::TensorNode* x_, TensorGraph::TensorNode* y_)
        : x(x_), y(y_)
    {
        inputs_ = {x};
        outputs_ = {y};
    }

    std::string op_name() const override { return "GELU"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGeluOp>(*this);
    }
};

//! GeLU activation: y = gelu(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Pointer to the output tensor
TensorGraph::TensorNode* gelu(
    TensorGraph::TensorNode* x,
    const std::string& output_name);

//! GeLU activation into pre-created output: y = gelu(x)
//! @param x Input tensor
//! @param y Output tensor (must already exist, same shape as x)
void gelu(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* y);

} // namespace nntile::graph::tensor
