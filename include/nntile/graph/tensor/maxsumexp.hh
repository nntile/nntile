/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/maxsumexp.hh
 * TensorGraph maxsumexp operation: (src, dst, axis)
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

//! MaxSumExp operation: dst = maxsumexp(src, axis)
struct TensorMaxsumexpOp : TensorGraph::OpNode
{
    Index axis;
    int redux;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorMaxsumexpOp() = default;
    TensorMaxsumexpOp(
        TensorGraph::TensorNode* src_,
        TensorGraph::TensorNode* dst_,
        Index axis_, int redux_)
        : axis(axis_), redux(redux_)
        , src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "MAXSUMEXP"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorMaxsumexpOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

TensorGraph::TensorNode* maxsumexp(
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index axis,
    int redux);

void maxsumexp(
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux);

} // namespace nntile::graph::tensor
