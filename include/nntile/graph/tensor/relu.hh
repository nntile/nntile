/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/relu.hh
 * TensorGraph ReLU operation: dst = relu(src)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! ReLU operation at tensor level: dst = relu(src)
struct TensorReluOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorReluOp() = default;
    TensorReluOp(TensorGraph::TensorNode* src_, TensorGraph::TensorNode* dst_)
        : src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "RELU"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorReluOp>(*this);
    }
};

//! ReLU: dst = relu(src) (creates output)
TensorGraph::TensorNode* relu(
    TensorGraph::TensorNode* src,
    const std::string& output_name);

//! ReLU: dst = relu(src) (uses existing output)
void relu(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
