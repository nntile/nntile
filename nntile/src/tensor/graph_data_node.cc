#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/graph_data_node.cc
 * Implementation of TensorGraph::TensorNode (include/nntile/tensor/graph_data_node.hh).
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/graph.hh"

#include <numeric>
#include <stdexcept>

namespace nntile
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
    member_index_.reserve(shape_.size());
    for(size_t i = 0; i < shape_.size(); ++i)
    {
        auto desc = std::make_shared<AxisDescriptor>();
        desc->extent = shape_[i];
        desc->members.push_back({static_cast<void*>(this),
                                  static_cast<int>(i)});
        member_index_.push_back(0);
        if(graph_ != nullptr)
        {
            graph_->note_axis_group(desc.get());
        }
        axes_.push_back(std::move(desc));
    }
}

void TensorGraph::TensorNode::note_member_index(int dim, std::size_t idx)
{
    if(dim < 0 || static_cast<size_t>(dim) >= member_index_.size())
    {
        throw std::out_of_range(
            "TensorNode::note_member_index: dim out of range");
    }
    member_index_[static_cast<size_t>(dim)] = idx;
}

void TensorGraph::TensorNode::unlink_from_axis_groups()
{
    for(size_t d = 0; d < axes_.size(); ++d)
    {
        std::shared_ptr<AxisDescriptor> const &desc = axes_[d];
        if(!desc)
        {
            continue;
        }
        std::size_t const idx = member_index_[d];
        auto &members = desc->members;
        if(idx >= members.size() || members[idx].first != this)
        {
            continue;
        }
        std::size_t const last = members.size() - 1;
        if(idx != last)
        {
            members[idx] = members[last];
            auto *other = static_cast<TensorNode *>(members[idx].first);
            int const other_d = members[idx].second;
            if(other != nullptr &&
                other_d >= 0 &&
                static_cast<size_t>(other_d) < other->member_index_.size())
            {
                other->member_index_[static_cast<size_t>(other_d)] = idx;
            }
        }
        members.pop_back();
        member_index_[d] = static_cast<std::size_t>(-1);
        if(members.empty() && graph_ != nullptr)
        {
            graph_->drop_axis_group(desc.get());
        }
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
        // Unify via merge_axis (union-by-size) instead of erasing this node
        // from old_members with a linear scan. Joining a fresh singleton into
        // a large shared group stays O(1); the old erase was O(|members|).
        std::shared_ptr<AxisDescriptor> other = axes[i];
        merge_axis(axes_[i], other);
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
    result += "], axes=[";
    for(size_t i = 0; i < axes_.size(); ++i)
    {
        if(i > 0) result += ", ";
        const auto& ax = axes_[i];
        if(!ax->name.empty())
        {
            result += ax->name;
        }
        else
        {
            result += std::to_string(ax->extent);
        }
        if(ax->is_tiled())
        {
            result += "/" + ax->tile_sizes_to_string();
        }
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

} // namespace nntile
