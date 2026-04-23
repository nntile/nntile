/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/add_slice_inplace.hh
 * TensorGraph add_slice_inplace operation: dst = alpha * src + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Add slice in-place at tensor level: dst = alpha * src + beta * dst
struct TensorAddSliceInplaceOp : TensorGraph::OpNode
{
    Index axis;
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorAddSliceInplaceOp() = default;
    TensorAddSliceInplaceOp(
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Scalar alpha_, Scalar beta_,
        Index axis_)
        : axis(axis_), alpha(alpha_), beta(beta_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "ADD_SLICE_INPLACE"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorAddSliceInplaceOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Add slice in-place: dst = alpha * src + beta * dst
void add_slice_inplace(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    Scalar beta,
    TensorGraph::TensorNode* dst,
    Index axis);

} // namespace nntile::graph::tensor
