/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/multiply_slice.hh
 * TensorGraph multiply_slice operation: dst = alpha * src * dst
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Multiply slice operation: dst = alpha * src * dst
struct TensorMultiplySliceOp : TensorGraph::OpNode
{
    Scalar alpha;
    Index axis;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorMultiplySliceOp() = default;
    TensorMultiplySliceOp(
        Scalar alpha_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_)
        : alpha(alpha_), axis(axis_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "MULTIPLY_SLICE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMultiplySliceOp>(*this);
    }
};

TensorGraph::TensorNode* multiply_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    Index axis_size);

void multiply_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis);

} // namespace nntile::graph::tensor
