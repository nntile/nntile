/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/ops/unregister.hh
 * TensorGraph async unregister (lowers to per-tile unregister_submit).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! Async unregister at tensor level (no outputs; side-effect on ``x``).
struct TensorUnregisterOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode *x = nullptr;

    TensorUnregisterOp() = default;
    explicit TensorUnregisterOp(TensorGraph::TensorNode *x_)
        : x(x_)
    {
        inputs_ = {x};
        outputs_ = {};
    }

    std::string op_name() const override { return "UNREGISTER"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorUnregisterOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void unregister(TensorGraph::TensorNode *x);

} // namespace nntile::tensor
