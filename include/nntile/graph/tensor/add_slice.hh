/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_slice.hh
 * TensorGraph add_slice operation: dst = alpha * src1 + beta * src2
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

//! Add slice operation at tensor level: dst = alpha * src1 + beta * src2
struct TensorAddSliceOp : TensorGraph::OpNode
{
    Index axis;
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorAddSliceOp() = default;
    TensorAddSliceOp(
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_,
        Scalar alpha_, Scalar beta_,
        Index axis_)
        : axis(axis_), alpha(alpha_), beta(beta_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "ADD_SLICE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddSliceOp>(*this);
    }
};

//! Add slice: dst = alpha * src1 + beta * src2 (creates output)
TensorGraph::TensorNode* add_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis);

//! Add slice: dst = alpha * src1 + beta * src2 (uses existing output)
void add_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis);

} // namespace nntile::graph::tensor
