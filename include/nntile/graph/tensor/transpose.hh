/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/transpose.hh
 * TensorGraph transpose operation: dst = alpha * transpose(src)
 *
 * @version 1.1.0
 * */

#pragma once

#include <string>

#include <nntile/base_types.hh>
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! Transpose operation at tensor level: dst = alpha * transpose(src)
struct TensorTransposeOp : TensorGraph::OpNode
{
    Index ndim = 0;
    Scalar alpha = 1.0;
    TensorGraph::TensorNode* src = nullptr;
    TensorGraph::TensorNode* dst = nullptr;

    TensorTransposeOp() = default;
    TensorTransposeOp(TensorGraph::TensorNode* src_,
                      TensorGraph::TensorNode* dst_,
                      Index ndim_, Scalar alpha_ = 1.0)
        : ndim(ndim_), alpha(alpha_), src(src_), dst(dst_)
    {
        inputs_ = {src};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "TRANSPOSE"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorTransposeOp>(*this);
    }
};

//! Transpose: dst = alpha * transpose(src) (creates output)
TensorGraph::TensorNode* transpose(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    Index ndim);

//! Transpose: dst = alpha * transpose(src) (uses existing output)
void transpose(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index ndim);

} // namespace nntile::graph
