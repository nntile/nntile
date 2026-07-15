/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/ops/maxsumexp.hh
 * TensorGraph maxsumexp operation: (src, dst, axis, beta)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! MaxSumExp operation: beta=0 overwrite dst; beta=1 accumulate
struct TensorMaxsumexpOp : TensorGraph::OpNode
{
    Index axis;
    Scalar beta;
    int redux;
    TensorGraph::TensorNode *src = nullptr;
    TensorGraph::TensorNode *dst = nullptr;

    TensorMaxsumexpOp() = default;
    TensorMaxsumexpOp(TensorGraph::TensorNode *src_,
        TensorGraph::TensorNode *dst_,
        Index axis_,
        Scalar beta_,
        int redux_) :
        axis(axis_), beta(beta_), redux(redux_), src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "MAXSUMEXP"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMaxsumexpOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

//! Create new dst and overwrite (beta=0 for first tile segment along axis)
TensorGraph::TensorNode *maxsumexp(
    TensorGraph::TensorNode *src, Index axis, int redux);

//! Write into existing dst: beta=0 overwrite, beta=1 accumulate
void maxsumexp(TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst,
    Index axis,
    Scalar beta,
    int redux);

} // namespace nntile::tensor
