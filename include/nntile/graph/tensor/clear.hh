/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/clear.hh
 * TensorGraph clear operation: x = 0
 *
 * @version 1.1.0
 * */

#pragma once

// NNTile headers
#include <nntile/graph/tensor/graph.hh>

namespace nntile::graph::tensor
{

//! Clear operation at tensor level: x = 0
struct TensorClearOp : TensorGraph::OpNode
{
    TensorGraph::TensorNode* x = nullptr;

    TensorClearOp() = default;
    explicit TensorClearOp(TensorGraph::TensorNode* x_)
        : x(x_)
    {
        inputs_ = {x};
        outputs_ = {x};
    }

    std::string op_name() const override { return "CLEAR"; }

    void execute(TensorGraph::Runtime& runtime) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorClearOp>(*this);
    }
};

//! Clear tensor: x = 0
void clear(TensorGraph::TensorNode* x);

} // namespace nntile::graph::tensor
