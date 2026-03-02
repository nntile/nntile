/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/base_op_node.hh
 * BaseOpNode - base class for graph operations, also serves as the graph node.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nntile/graph/execution_context.hh>

namespace nntile::graph
{

//! Base class for all graph operations. Also serves as the graph node.
//! Each operation stores its parameters, inputs, outputs, graph reference, and id.
//! Dispatch is via virtual execute(); no OpType enum.
//! @tparam Graph The graph type (e.g. TensorGraph)
//! @tparam DataNode The data node type (e.g. TensorGraphNode)
template<typename Graph, typename DataNode>
class BaseOpNode
{
public:
    using NodeId = uint64_t;

    virtual ~BaseOpNode() = default;

    //! Operation type name for visualization (e.g. "ADD", "GEMM")
    virtual std::string op_name() const = 0;

    //! Optional node name (for debugging/visualization)
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    //! Graph node id (assigned when added to graph)
    NodeId id() const { return id_; }

    //! Input data nodes (owned by the graph; op holds pointers)
    const std::vector<DataNode*>& inputs() const { return inputs_; }

    //! Output data nodes (owned by the graph; op holds pointers)
    const std::vector<DataNode*>& outputs() const { return outputs_; }

    //! Execute this operation. Uses inputs/outputs from this op;
    //! gets runtime data from the context mapping.
    virtual void execute(ExecutionContext<DataNode>& ctx) const = 0;

    //! Optional: clone for graph copying
    virtual std::shared_ptr<BaseOpNode<Graph, DataNode>> clone() const = 0;

protected:
    BaseOpNode() = default;

    NodeId id_ = -1;
    std::string name_;
    std::vector<DataNode*> inputs_;
    std::vector<DataNode*> outputs_;

    friend Graph;  // Graph assigns id_ when adding
};

} // namespace nntile::graph
