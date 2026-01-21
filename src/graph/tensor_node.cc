/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor_node.cc
 * Implementation of TensorNode class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/tensor_node.hh"

// Include standard headers
#include <algorithm>
#include <string>

// Include third-party headers

// Include other NNTile headers

namespace nntile::graph
{

//! A tensor node in the logical graph
TensorNode::TensorNode(
    NodeId id,
    const std::string& name,
    TensorSpec spec,
    LogicalGraph* graph)
    : id_(id)
    , name_(name)
    , spec_(std::move(spec))
    , graph_(graph)
{
}

//! String representation
std::string TensorNode::to_string() const
{
    return "TensorNode(id=" + std::to_string(id_) + ", name='" + name_ +
        "', " + spec_.to_string() + ")";
}

//! Remove a consumer from this tensor's consumer list
void TensorNode::remove_consumer(OpNode* op)
{
    auto it = std::find(consumers_.begin(), consumers_.end(), op);
    if(it != consumers_.end())
    {
        consumers_.erase(it);
    }
}

} // namespace nntile::graph
