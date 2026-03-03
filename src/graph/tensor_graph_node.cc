/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/graph/tensor_graph_node.cc
 * TensorGraph::TensorNode implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/graph.hh"

namespace nntile::graph
{

TensorGraph::TensorNode::TensorNode(
    NodeId id,
    TensorGraph* graph,
    std::vector<Index> shape,
    DataType dtype,
    const std::string& name)
    : id_(id)
    , graph_(graph)
    , shape_(std::move(shape))
    , dtype_(dtype)
    , name_(name)
{
    for(Index dim : shape_)
    {
        if(dim <= 0)
        {
            throw std::invalid_argument(
                "TensorGraph::TensorNode: all dimensions must be positive");
        }
    }
}

Index TensorGraph::TensorNode::dim(int idx) const
{
    if(idx < 0)
    {
        idx += static_cast<int>(shape_.size());
    }
    if(idx < 0 || static_cast<size_t>(idx) >= shape_.size())
    {
        throw std::out_of_range("TensorGraph::TensorNode::dim: index out of range");
    }
    return shape_[static_cast<size_t>(idx)];
}

Index TensorGraph::TensorNode::nelems() const
{
    return std::accumulate(shape_.begin(), shape_.end(), Index(1),
        std::multiplies<Index>());
}

size_t TensorGraph::TensorNode::size_bytes() const
{
    return static_cast<size_t>(nelems()) * dtype_size(dtype_);
}

bool TensorGraph::TensorNode::is_compatible(const TensorNode* other) const
{
    return other != nullptr && dtype_ == other->dtype_;
}

TensorGraph* TensorGraph::TensorNode::graph()
{
    return graph_;
}

const TensorGraph* TensorGraph::TensorNode::graph() const
{
    return graph_;
}

std::string TensorGraph::TensorNode::to_string() const
{
    std::string result = "TensorGraph::TensorNode(id=" +
        std::to_string(id_) + ", name='" + name_ + "', shape=[";
    for(size_t i = 0; i < shape_.size(); ++i)
    {
        if(i > 0) result += ", ";
        result += std::to_string(shape_[i]);
    }
    result += "], dtype=" + dtype_to_string(dtype_) + ")";
    return result;
}

} // namespace nntile::graph
