/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/silu_inplace.hh
 * TensorGraph silu_inplace operation: dst = silu(dst)
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph
{
struct LoweringContext;
}

namespace nntile::graph::tensor
{

//! SiLU in-place operation: dst = silu(dst)
struct TensorSiluInplaceOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* dst = nullptr;

    TensorSiluInplaceOp() = default;
    TensorSiluInplaceOp(TensorGraph::TensorNode* dst_)
        : dst(dst_)
    {
        inputs_ = {dst};
        outputs_ = {dst};
    }

    std::string op_name() const override { return "SILU_INPLACE"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorSiluInplaceOp>(*this);
    }

    void lower_to_tile(const LoweringContext& ctx) const override;
};

void silu_inplace(TensorGraph::TensorNode* dst);

} // namespace nntile::graph::tensor
