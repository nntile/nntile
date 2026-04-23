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

// Standard library headers
#include <memory>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/graph/tensor/graph_data_node.hh>

namespace nntile::graph
{

struct LoweringContext;

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

    virtual std::shared_ptr<TensorGraph::OpNode> clone() const = 0;

    //! Lower this op into tile ops on ctx.out. Default throws; ops that support
    //! tile lowering override (same pattern as execute()).
    virtual void lower_to_tile(const LoweringContext& ctx) const;

protected:
    OpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    std::vector<TensorGraph::TensorNode*> inputs_;
    std::vector<TensorGraph::TensorNode*> outputs_;

    friend class TensorGraph;
};

} // namespace nntile::graph
