/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/gelutanh.hh
 * TensorGraph gelutanh operation: dst = gelutanh(src)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! GeLUTanh operation: dst = gelutanh(src)
struct TensorGelutanhOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorGelutanhOp() = default;
    TensorGelutanhOp(TensorGraph::TensorNode* src_, TensorGraph::TensorNode* dst_)
        : src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "GELUTANH"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGelutanhOp>(*this);
    }
};

TensorGraph::TensorNode* gelutanh(
    TensorGraph::TensorNode* src,
    const std::string& output_name);

void gelutanh(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph
