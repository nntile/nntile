/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/softmax.hh
 * TensorGraph softmax operation: (maxsumexp, src, dst, axis)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Softmax operation: dst = softmax(maxsumexp, src, alpha, axis)
struct TensorSoftmaxOp : TensorGraph::OpNode
{
    Scalar alpha;
    Index axis;
    TensorGraph::TensorNode* maxsumexp = nullptr;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorSoftmaxOp() = default;
    TensorSoftmaxOp(
        TensorGraph::TensorNode* maxsumexp_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Scalar alpha_,
        Index axis_)
        : alpha(alpha_), axis(axis_)
        , maxsumexp(maxsumexp_), src(src_), dst(dst_)
    {
        inputs_ = {maxsumexp, src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SOFTMAX"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSoftmaxOp>(*this);
    }
};

TensorGraph::TensorNode* softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Scalar alpha,
    Index axis);

void softmax(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Scalar alpha,
    Index axis);

} // namespace nntile::graph
