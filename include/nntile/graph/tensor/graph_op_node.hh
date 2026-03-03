/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor/graph_op_node.hh
 * TensorGraph::OpNode - base class for TensorGraph operations.
 *
 * @version 1.1.0
 * */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nntile/graph/tensor/graph_data_node.hh>
#include <nntile/graph/tensor/graph_runtime.hh>

namespace nntile::graph
{

//! Base class for TensorGraph operations. Each op stores inputs, outputs, id.
//! Dispatch is via virtual execute(); no OpType enum.
class TensorGraph::OpNode
{
public:
    using NodeId = uint64_t;

    virtual ~OpNode() = default;

    virtual std::string op_name() const = 0;
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    NodeId id() const { return id_; }

    const std::vector<TensorGraph::TensorNode*>& inputs() const
    {
        return inputs_;
    }
    const std::vector<TensorGraph::TensorNode*>& outputs() const
    {
        return outputs_;
    }

    virtual void execute(TensorGraph::Runtime& runtime) const = 0;
    virtual std::shared_ptr<TensorGraph::OpNode> clone() const = 0;

protected:
    OpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    std::vector<TensorGraph::TensorNode*> inputs_;
    std::vector<TensorGraph::TensorNode*> outputs_;

    friend class TensorGraph;
};

} // namespace nntile::graph
