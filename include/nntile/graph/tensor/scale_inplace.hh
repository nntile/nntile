/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/scale_inplace.hh
 * TensorGraph scale_inplace operation: dst = alpha * dst
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

//! Scale in-place operation at tensor level: dst = alpha * dst
struct TensorScaleInplaceOp : TensorGraph::OpNode
{
    Scalar alpha;
    TensorGraph::TensorNode* dst = nullptr;

    TensorScaleInplaceOp() = default;
    TensorScaleInplaceOp(Scalar alpha_, TensorGraph::TensorNode* dst_)
        : alpha(alpha_), dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SCALE_INPLACE"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorScaleInplaceOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Scale in-place: dst = alpha * dst
void scale_inplace(Scalar alpha, TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
