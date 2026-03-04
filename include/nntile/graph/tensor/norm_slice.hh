/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/norm_slice.hh
 * TensorGraph norm_slice operation
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Norm slice operation: dst = alpha*norm(src1) + beta*src2
struct TensorNormSliceOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    Index axis;
    int redux;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorNormSliceOp() = default;
    TensorNormSliceOp(
        Scalar alpha_, Scalar beta_,
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_)
        : alpha(alpha_), beta(beta_)
        , axis(axis_), redux(redux_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "NORM_SLICE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorNormSliceOp>(*this);
    }
};

TensorGraph::TensorNode* norm_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis,
    int redux);

void norm_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux);

} // namespace nntile::graph::tensor
