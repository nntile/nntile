/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tensor/ops/invalidate.hh
 * TensorGraph async invalidate (lowers to per-tile invalidate_submit).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstddef>

#include <nntile/tensor/graph.hh>

namespace nntile
{
struct LoweringContext;
}

namespace nntile::tensor
{

//! Async invalidate at tensor level (no outputs; side-effect on ``x``).
struct TensorInvalidateOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode *x = nullptr;

    TensorInvalidateOp() = default;
    explicit TensorInvalidateOp(TensorGraph::TensorNode *x_)
        : x(x_)
    {
        inputs_ = {x};
        outputs_ = {};
    }

    std::string op_name() const override { return "INVALIDATE"; }

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorInvalidateOp>(*this);
    }

    void lower_to_tile(const LoweringContext &ctx) const override;
};

void invalidate(TensorGraph::TensorNode *x);

//! For every tensor touched by unsealed ops that has no live ``TensorRef``,
//! append ``INVALIDATE``. O(phase). Call before ``seal_phase()``.
std::size_t append_invalidates_for_unmarked_unsealed(TensorGraph &graph);

} // namespace nntile::tensor
