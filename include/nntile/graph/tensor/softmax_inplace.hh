/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/softmax_inplace.hh
 * TensorGraph softmax_inplace operation: (maxsumexp, alpha, dst, axis)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Softmax in-place: dst = softmax(maxsumexp, alpha, dst, axis)
struct TensorSoftmaxInplaceOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    Index axis = 0;
    TensorGraph::TensorNode* maxsumexp = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorSoftmaxInplaceOp() = default;
    TensorSoftmaxInplaceOp(
        TensorGraph::TensorNode* maxsumexp_,
        TensorGraph::TensorNode* dst_,
        Scalar alpha_ = 1.0,
        Index axis_ = 0)
        : alpha(alpha_), axis(axis_)
        , maxsumexp(maxsumexp_), dst(dst_)
    {
        inputs_ = {maxsumexp, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SOFTMAX_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSoftmaxInplaceOp>(*this);
    }
};

void softmax_inplace(
    TensorGraph::TensorNode* maxsumexp,
    TensorGraph::TensorNode* dst,
    Scalar alpha = 1.0,
    Index axis = 0);

} // namespace nntile::graph
