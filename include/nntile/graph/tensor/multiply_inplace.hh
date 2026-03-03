/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/multiply_inplace.hh
 * TensorGraph multiply_inplace operation: dst = alpha * src * dst
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Multiply in-place at tensor level: dst = alpha * src * dst
struct TensorMultiplyInplaceOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorMultiplyInplaceOp() = default;
    TensorMultiplyInplaceOp(TensorGraph::TensorNode* src_,
                            TensorGraph::TensorNode* dst_,
                            Scalar alpha_ = 1.0)
        : alpha(alpha_), src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "MULTIPLY_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMultiplyInplaceOp>(*this);
    }
};

//! Multiply in-place: dst = alpha * src * dst
void multiply_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph
