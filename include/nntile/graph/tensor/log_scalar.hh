/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/log_scalar.hh
 * TensorGraph log_scalar: log scalar value from tensor (debugging/monitoring)
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor/graph.hh>
#include <string>

namespace nntile::graph
{

//! Log scalar: log scalar value from tensor
struct TensorLogScalarOp : TensorGraph::OpNode
{
    std::string name;
    TensorGraph::TensorNode* value = nullptr;

    TensorLogScalarOp() = default;
    TensorLogScalarOp(const std::string& name_,
                     TensorGraph::TensorNode* value_)
        : name(name_), value(value_)
    {
        inputs_ = {value};
        outputs_ = {};
    }

    std::string op_name() const override { return "LOG_SCALAR"; }

    void execute(TensorGraph::ExecutionContext& ctx) const override;

    std::shared_ptr<TensorGraph::OpNode> clone() const override
    {
        return std::make_shared<TensorLogScalarOp>(*this);
    }
};

//! Log scalar: log scalar value from tensor
void log_scalar(const std::string& name,
                TensorGraph::TensorNode* value);

} // namespace nntile::graph
