/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/graph_data_node.cc
 * Implementation of TileGraph::TileNode.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/graph.hh"

#include <numeric>
#include <stdexcept>

namespace nntile::graph
{

TileGraph::TileNode::TileNode(
    NodeId id,
    TileGraph* graph,
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
                "TileGraph::TileNode: all dimensions must be positive");
        }
    }
}

Index TileGraph::TileNode::dim(int idx) const
{
    if(idx < 0)
    {
        idx += static_cast<int>(shape_.size());
    }
    if(idx < 0 || static_cast<size_t>(idx) >= shape_.size())
    {
        throw std::out_of_range("TileGraph::TileNode::dim: index out of range");
    }
    return shape_[static_cast<size_t>(idx)];
}

Index TileGraph::TileNode::nelems() const
{
    return std::accumulate(shape_.begin(), shape_.end(), Index(1),
        std::multiplies<Index>());
}

size_t TileGraph::TileNode::size_bytes() const
{
    return static_cast<size_t>(nelems()) * dtype_size(dtype_);
}

TileGraph* TileGraph::TileNode::graph()
{
    return graph_;
}

const TileGraph* TileGraph::TileNode::graph() const
{
    return graph_;
}

void TileGraph::TileNode::set_tensor_info(
    TileGraph::TensorDescriptor* desc,
    std::vector<Index> coord)
{
    tensor_desc_ = desc;
    tile_coord_ = std::move(coord);
}

std::string TileGraph::TileNode::to_string() const
{
    std::string result = "TileGraph::TileNode(id=" +
        std::to_string(id_) + ", name='" + name_ + "', shape=[";
    for(size_t i = 0; i < shape_.size(); ++i)
    {
        if(i > 0) result += ", ";
        result += std::to_string(shape_[i]);
    }
    result += "], dtype=" + dtype_to_string(dtype_);
    if(tensor_desc_ != nullptr)
    {
        result += ", tensor='" + tensor_desc_->tensor_name + "', coord=[";
        for(size_t i = 0; i < tile_coord_.size(); ++i)
        {
            if(i > 0) result += ", ";
            result += std::to_string(tile_coord_[i]);
        }
        result += "]";
    }
    result += ")";
    return result;
}

void TileGraph::TileNode::set_bind_hint(std::vector<std::uint8_t> data)
{
    const size_t expected = size_bytes();
    if(data.size() != expected)
    {
        throw std::invalid_argument(
            "TileGraph::TileNode::set_bind_hint: size mismatch, expected " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(data.size()));
    }
    bind_hint_.emplace(std::move(data));
}

const std::vector<std::uint8_t>* TileGraph::TileNode::get_bind_hint() const
{
    if(bind_hint_.has_value())
    {
        return &(*bind_hint_);
    }
    return nullptr;
}

} // namespace nntile::graph
