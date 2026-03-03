/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/sumprod_slice.hh
 * TensorGraph sumprod_slice operation: dst = alpha * sumprod_slice(src1, src2) + beta * dst
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Sumprod slice operation: dst = alpha * sumprod_slice(src1, src2) + beta * dst
struct TensorSumprodSliceOp : TensorGraph::OpNode
{
    Index axis = 0;
    int redux = 0;
    Scalar alpha = 1.0;
    Scalar beta = 0.0;
    TensorGraph::TensorNode* src1 = nullptr;
    TensorGraph::TensorNode* src2 = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorSumprodSliceOp() = default;
    TensorSumprodSliceOp(
        TensorGraph::TensorNode* src1_,
        TensorGraph::TensorNode* src2_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_,
        Scalar alpha_, Scalar beta_)
        : axis(axis_), redux(redux_), alpha(alpha_), beta(beta_)
        , src1(src1_), src2(src2_), dst(dst_)
    {
        inputs_ = {src1, src2, dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SUMPROD_SLICE"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSumprodSliceOp>(*this);
    }
};

//! Sumprod over fibers into slice: dst = alpha * sumprod_slice(src1, src2) + beta * dst
void sumprod_slice(
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis = 0,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0);

} // namespace nntile::graph
