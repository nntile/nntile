/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/sum_slice.hh
 * TensorGraph sum_slice operation: dst = alpha * sum_slice(src) + beta * dst
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

//! Sum slice operation at tensor level: dst = alpha * sum_slice(src) + beta * dst
struct TensorSumSliceOp : TensorGraph::OpNode
{
    Index axis;
    int redux;
    Scalar alpha;
    Scalar beta;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorSumSliceOp() = default;
    TensorSumSliceOp(
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_,
        Scalar alpha_, Scalar beta_)
        : axis(axis_), redux(redux_), alpha(alpha_), beta(beta_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SUM_SLICE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSumSliceOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

//! Sum over fibers into slice: dst = alpha * sum_slice(src) + beta * dst (creates output)
TensorGraph::TensorNode* sum_slice(
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta);

//! Sum over fibers into slice: dst = alpha * sum_slice(src) + beta * dst (uses existing output)
void sum_slice(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta);

} // namespace nntile::graph::tensor
