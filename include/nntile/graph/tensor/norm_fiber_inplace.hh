/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/norm_fiber_inplace.hh
 * TensorGraph norm_fiber_inplace operation
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Norm fiber in-place: dst = alpha*norm(src) + beta*dst
struct TensorNormFiberInplaceOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    Scalar beta = 1.0;
    Index axis = 0;
    Index batch_ndim = 0;
    int redux = 0;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorNormFiberInplaceOp() = default;
    TensorNormFiberInplaceOp(
        Scalar alpha_, Scalar beta_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_, Index batch_ndim_, int redux_ = 0)
        : alpha(alpha_), beta(beta_)
        , axis(axis_), batch_ndim(batch_ndim_), redux(redux_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "NORM_FIBER_INPLACE"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorNormFiberInplaceOp>(*this);
    }
};

void norm_fiber_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim = 0,
    int redux = 0);

} // namespace nntile::graph
