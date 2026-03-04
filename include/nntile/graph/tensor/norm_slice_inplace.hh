/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/norm_slice_inplace.hh
 * TensorGraph norm_slice_inplace operation
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Norm slice in-place: dst = alpha*norm(src) + beta*dst
struct TensorNormSliceInplaceOp : TensorGraph::OpNode
{
    Scalar alpha;
    Scalar beta;
    Index axis;
    int redux;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorNormSliceInplaceOp() = default;
    TensorNormSliceInplaceOp(
        Scalar alpha_, Scalar beta_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_)
        : alpha(alpha_), beta(beta_)
        , axis(axis_), redux(redux_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "NORM_SLICE_INPLACE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorNormSliceInplaceOp>(*this);
    }
};

void norm_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux);

} // namespace nntile::graph::tensor
