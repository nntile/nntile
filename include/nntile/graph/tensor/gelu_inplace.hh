/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/gelu_inplace.hh
 * TensorGraph gelu_inplace operation: dst = gelu(dst)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{

//! GeLU in-place operation: dst = gelu(dst)
struct TensorGeluInplaceOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* dst = nullptr;

    TensorGeluInplaceOp() = default;
    TensorGeluInplaceOp(TensorGraph::TensorNode* dst_)
        : dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "GELU_INPLACE"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorGeluInplaceOp>(*this);
    }
};

void gelu_inplace(TensorGraph::TensorNode* dst);

} // namespace nntile::graph
