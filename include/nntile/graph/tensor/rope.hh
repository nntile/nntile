/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/rope.hh
 * TensorGraph rope operation: (sin, cos, src, dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! RoPE operation: dst = rope(sin, cos, src)
struct TensorRopeOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* sin = nullptr;
    TensorGraph::TensorNode* cos = nullptr;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorRopeOp() = default;
    TensorRopeOp(
        TensorGraph::TensorNode* sin_,
        TensorGraph::TensorNode* cos_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_)
        : sin(sin_), cos(cos_), src(src_), dst(dst_)
    {
        inputs_ = {sin, cos, src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "ROPE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorRopeOp>(*this);
    }
};

TensorGraph::TensorNode* rope(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* src,
    const std::string& output_name);

void rope(
    TensorGraph::TensorNode* sin,
    TensorGraph::TensorNode* cos,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
