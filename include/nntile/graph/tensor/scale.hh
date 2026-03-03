/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/scale.hh
 * TensorGraph scale operation: dst = alpha * src
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Scale operation at tensor level: dst = alpha * src
struct TensorScaleOp : TensorGraph::OpNode
{
    Scalar alpha = 1.0;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorScaleOp() = default;
    TensorScaleOp(TensorGraph::TensorNode* src_,
                 TensorGraph::TensorNode* dst_,
                 Scalar alpha_ = 1.0)
        : alpha(alpha_), src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SCALE"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorScaleOp>(*this);
    }
};

//! Scale operation: dst = alpha * src (creates output)
//! @param alpha Scaling factor
//! @param src Input tensor
//! @param output_name Name for the output tensor
//! @return Pointer to the output tensor
TensorGraph::TensorNode* scale(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name);

//! Scale operation: dst = alpha * src (uses existing output)
void scale(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst);

} // namespace nntile::graph
