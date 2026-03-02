/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/base_data_node.hh
 * BaseDataNode<Graph> - represents a data node in a graph (e.g. TensorGraph,
 * TileGraph).
 *
 * @version 1.1.0
 * */

#pragma once

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdint>

#include <nntile/base_types.hh>
#include <nntile/graph/dtype.hh>

namespace nntile::graph
{

//! Base data node - represents a tensor/tile in a graph.
//! @tparam Graph The graph type (e.g. TensorGraph, TileGraph)
template<typename Graph>
class BaseDataNode
{
public:
    using NodeId = uint64_t;

    BaseDataNode(
        NodeId id,
        Graph* graph,
        std::vector<Index> shape,
        DataType dtype,
        const std::string& name = ""
    );

    // Accessors
    NodeId id() const { return id_; }
    const std::string& name() const { return name_; }
    DataType dtype() const { return dtype_; }
    const std::vector<Index>& shape() const { return shape_; }
    Index ndim() const { return static_cast<Index>(shape_.size()); }
    Index dim(int idx) const;
    Index nelems() const;
    size_t size_bytes() const;
    bool is_compatible(const BaseDataNode* other) const;

    // Graph access
    Graph* graph() { return graph_; }
    const Graph* graph() const { return graph_; }

    // Graph structure
    bool is_input() const { return is_input_; }
    bool is_output() const { return is_output_; }
    void mark_input(bool v = true) { is_input_ = v; }
    void mark_output(bool v = true) { is_output_ = v; }

    // String representation
    std::string to_string() const;

private:
    NodeId id_;
    Graph* graph_;
    std::vector<Index> shape_;
    DataType dtype_;
    std::string name_;
    bool is_input_ = false;
    bool is_output_ = false;

    friend Graph;
};

// -------------------------------------------------------------------------
// Template implementations
// -------------------------------------------------------------------------

template<typename Graph>
BaseDataNode<Graph>::BaseDataNode(
    NodeId id,
    Graph* graph,
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
                "BaseDataNode: all dimensions must be positive");
        }
    }
}

template<typename Graph>
Index BaseDataNode<Graph>::dim(int idx) const
{
    if(idx < 0)
    {
        idx += static_cast<int>(shape_.size());
    }
    if(idx < 0 || static_cast<size_t>(idx) >= shape_.size())
    {
        throw std::out_of_range("BaseDataNode::dim: index out of range");
    }
    return shape_[static_cast<size_t>(idx)];
}

template<typename Graph>
Index BaseDataNode<Graph>::nelems() const
{
    return std::accumulate(shape_.begin(), shape_.end(), Index(1),
        std::multiplies<Index>());
}

template<typename Graph>
size_t BaseDataNode<Graph>::size_bytes() const
{
    return static_cast<size_t>(nelems()) * dtype_size(dtype_);
}

template<typename Graph>
bool BaseDataNode<Graph>::is_compatible(const BaseDataNode* other) const
{
    return other != nullptr && dtype_ == other->dtype_;
}

template<typename Graph>
std::string BaseDataNode<Graph>::to_string() const
{
    std::string result = "BaseDataNode(id=" +
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
