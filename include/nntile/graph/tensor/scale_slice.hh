/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/scale_slice.hh
 * TensorGraph scale_slice operation: dst = alpha * src (slice broadcast)
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <string>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! Scale slice operation: dst = alpha * src (slice broadcast)
struct TensorScaleSliceOp : TensorGraph::OpNode
{
    Scalar alpha;
    Index axis;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorScaleSliceOp() = default;
    TensorScaleSliceOp(
        Scalar alpha_,
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_)
        : alpha(alpha_), axis(axis_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SCALE_SLICE"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorScaleSliceOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

TensorGraph::TensorNode* scale_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    Index axis_size);

void scale_slice(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis);

} // namespace nntile::graph::tensor
