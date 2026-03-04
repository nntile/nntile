/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/copy.hh
 * TensorGraph copy operation: dst = src
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Copy operation at tensor level: dst = src
struct TensorCopyOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorCopyOp() = default;
    TensorCopyOp(TensorGraph::TensorNode* src_, TensorGraph::TensorNode* dst_)
        : src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "COPY"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorCopyOp>(*this);
    }
};

//! Copy: dst = src (creates output)
TensorGraph::TensorNode* copy(
    TensorGraph::TensorNode* src,
    const std::string& output_name);

//! Copy: dst = src (uses existing output)
void copy(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
