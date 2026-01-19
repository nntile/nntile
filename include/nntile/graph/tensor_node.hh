/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tensor_node.hh
 * TensorNode class for logical graph tensor nodes.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/graph/tensor_spec.hh>
#include <string>
#include <vector>
#include <cstdint>

namespace nntile::graph
{

// Forward declarations
class OpNode;
class LogicalGraph;

//! Unique identifier for nodes
using NodeId = uint64_t;

//! A tensor node in the logical graph
class TensorNode {
    friend class LogicalGraph;
    friend class OpNode;

private:
    NodeId id_;
    std::string name_;
    TensorSpec spec_;
    LogicalGraph* graph_;

    // Graph edges
    OpNode* producer_ = nullptr;           // Op that creates this tensor (nullptr if input)
    std::vector<OpNode*> consumers_;       // Ops that use this tensor

public:
    TensorNode(NodeId id, const std::string& name, TensorSpec spec, LogicalGraph* graph);

    // Accessors
    NodeId id() const { return id_; }
    const std::string& name() const { return name_; }
    const TensorSpec& spec() const { return spec_; }
    DataType dtype() const { return spec_.dtype(); }
    const std::vector<Index>& shape() const { return spec_.shape(); }
    Index ndim() const { return spec_.ndim(); }

    // Graph structure
    bool has_producer() const { return producer_ != nullptr; }
    OpNode* producer() const { return producer_; }
    const std::vector<OpNode*>& consumers() const { return consumers_; }

    // String representation
    std::string to_string() const;

private:
    // Only LogicalGraph/OpNode can modify edges
    void set_producer(OpNode* op) { producer_ = op; }
    void add_consumer(OpNode* op) { consumers_.push_back(op); }
};

} // namespace nntile::graph
