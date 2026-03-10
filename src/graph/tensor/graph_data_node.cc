/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/graph_data_node.cc
 * Implementation of TensorGraph::TensorNode (include/nntile/graph/tensor/graph_data_node.hh).
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/graph.hh"

#include <algorithm>
#include <numeric>
#include <stdexcept>

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

    axes_.reserve(shape_.size());
    for(size_t i = 0; i < shape_.size(); ++i)
    {
        auto desc = std::make_shared<AxisDescriptor>();
        desc->extent = shape_[i];
        desc->members.push_back({static_cast<void*>(this),
                                  static_cast<int>(i)});
        axes_.push_back(std::move(desc));
    }
}

void TensorGraph::TensorNode::set_axes(
    const std::vector<std::shared_ptr<AxisDescriptor>>& axes)
{
    if(axes.size() != axes_.size())
    {
        throw std::invalid_argument(
            "TensorNode::set_axes: axes size mismatch");
    }
    for(size_t i = 0; i < axes.size(); ++i)
    {
        if(axes[i]->extent != shape_[i])
        {
            throw std::invalid_argument(
                "TensorNode::set_axes: extent mismatch at axis " +
                std::to_string(i));
        }
        axes_[i]->members.clear();
        axes_[i] = axes[i];
        axes[i]->members.push_back(
            {static_cast<void*>(this), static_cast<int>(i)});
    }
}

AxisDescriptor* TensorGraph::TensorNode::axis(int i) const
{
    if(i < 0)
    {
        i += static_cast<int>(axes_.size());
    }
    if(i < 0 || static_cast<size_t>(i) >= axes_.size())
    {
        throw std::out_of_range(
            "TensorGraph::TensorNode::axis: index out of range");
    }
    return axes_[static_cast<size_t>(i)].get();
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

void TensorGraph::TensorNode::set_bind_hint(std::vector<std::uint8_t> data)
{
    const size_t expected = size_bytes();
    if(data.size() != expected)
    {
        throw std::invalid_argument(
            "TensorGraph::TensorNode::set_bind_hint: size mismatch, expected " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(data.size()));
    }
    bind_hint_.emplace(std::move(data));
}

const std::vector<std::uint8_t>* TensorGraph::TensorNode::get_bind_hint() const
{
    if(bind_hint_.has_value())
    {
        return &(*bind_hint_);
    }
    return nullptr;
}

} // namespace nntile::graph
