/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_op_node.hh
 * TensorGraphOpNode - base class for TensorGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nntile/graph/execution_context.hh>
#include <nntile/graph/tensor/graph_tensor_node.hh>

namespace nntile::graph
{

//! Base class for TensorGraph operations. Each op stores inputs, outputs, id.
//! Dispatch is via virtual execute(); no OpType enum.
class TensorGraphOpNode
{
public:
    using NodeId = uint64_t;

    virtual ~TensorGraphOpNode() = default;

    virtual std::string op_name() const = 0;
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    NodeId id() const { return id_; }

    const std::vector<TensorGraphNode*>& inputs() const { return inputs_; }
    const std::vector<TensorGraphNode*>& outputs() const { return outputs_; }

    virtual void execute(ExecutionContext<TensorGraphNode>& ctx) const = 0;
    virtual std::shared_ptr<TensorGraphOpNode> clone() const = 0;

protected:
    TensorGraphOpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    std::vector<TensorGraphNode*> inputs_;
    std::vector<TensorGraphNode*> outputs_;

    friend class TensorGraph;
};

} // namespace nntile::graph
