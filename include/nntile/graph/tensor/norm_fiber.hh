/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/norm_fiber.hh
 * TensorGraph norm_fiber operation
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Norm fiber operation: dst = alpha*norm(src1) + beta*src2
struct TensorNormFiberOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    Index axis;
    Index batch_ndim;
    int redux;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorNormFiberOp() = default;
    TensorNormFiberOp(
        Scalar alpha_, Scalar beta_,
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_,
        Index axis_, Index batch_ndim_, int redux_)
        : alpha(alpha_), beta(beta_)
        , axis(axis_), batch_ndim(batch_ndim_), redux(redux_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "NORM_FIBER"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorNormFiberOp>(*this);
    }
};

TensorGraph::TensorNode* norm_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    const std::string& output_name,
    Index axis,
    Index batch_ndim,
    int redux);

void norm_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src1,
    Scalar beta,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim,
    int redux);

} // namespace nntile::graph
