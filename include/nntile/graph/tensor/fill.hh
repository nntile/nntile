/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/fill.hh
 * TensorGraph fill operation: x = val
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

//! Fill operation at tensor level: x = val
struct TensorFillOp : TensorGraph::OpNode
{
    Scalar val;
    TensorGraph::TensorNode* x = nullptr;

    TensorFillOp() = default;
    TensorFillOp(TensorGraph::TensorNode* x_, Scalar val_)
        : val(val_), x(x_)
    {
        inputs_ = {x};
        outputs_ = {x};
    }

    std::string op_name() const override { return "FILL"; }


    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorFillOp>(*this);
    }
    void lower_to_tile(const LoweringContext& ctx) const override;

};

//! Fill tensor: x = val
void fill(Scalar val, TensorGraph::TensorNode* x);

} // namespace nntile::graph::tensor
