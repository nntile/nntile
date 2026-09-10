#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/axis_descriptor.cc
 * AxisDescriptor and merge_axis implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/graph.hh"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace nntile
{

void AxisDescriptor::set_tiling(Index tile_size)
{
    if(tile_size <= 0)
    {
        throw std::invalid_argument(
            "AxisDescriptor::set_tiling: tile_size must be positive");
    }
    tile_sizes.clear();
    for(Index remaining = extent; remaining > 0; remaining -= tile_size)
    {
        tile_sizes.push_back(std::min(tile_size, remaining));
    }
}

void AxisDescriptor::set_tiling(const std::vector<Index>& sizes)
{
    if(sizes.empty())
    {
        throw std::invalid_argument(
            "AxisDescriptor::set_tiling: sizes must be non-empty");
    }
    Index total = std::accumulate(sizes.begin(), sizes.end(), Index(0));
    if(total != extent)
    {
        throw std::invalid_argument(
            "AxisDescriptor::set_tiling: sum of tile sizes (" +
            std::to_string(total) + ") must equal extent (" +
            std::to_string(extent) + ")");
    }
    for(Index s : sizes)
    {
        if(s <= 0)
        {
            throw std::invalid_argument(
                "AxisDescriptor::set_tiling: all tile sizes must be positive");
        }
    }
    tile_sizes = sizes;
}

Index AxisDescriptor::num_tiles() const
{
    if(tile_sizes.empty())
    {
        return 1;
    }
    return static_cast<Index>(tile_sizes.size());
}

std::string AxisDescriptor::tile_sizes_to_string() const
{
    if(tile_sizes.empty())
    {
        return "";
    }
    if(tile_sizes.size() == 1)
    {
        return std::to_string(tile_sizes[0]);
    }
    // Base+leftover pattern (from set_tiling(Index)): all except last equal
    // base; last tile may be smaller. Treat as uniform "N" like
    // compute_basetile_shape.
    Index base = tile_sizes[0];
    bool base_plus_leftover = true;
    for(size_t t = 1; t < tile_sizes.size() - 1; ++t)
    {
        if(tile_sizes[t] != base)
        {
            base_plus_leftover = false;
            break;
        }
    }
    Index last = tile_sizes.back();
    if(base_plus_leftover && last > 0 && last <= base)
    {
        return std::to_string(base);
    }
    std::string result = "{";
    for(size_t t = 0; t < tile_sizes.size(); ++t)
    {
        if(t > 0) result += ",";
        result += std::to_string(tile_sizes[t]);
    }
    result += "}";
    return result;
}

void merge_axis(std::shared_ptr<AxisDescriptor>& lhs,
                std::shared_ptr<AxisDescriptor>& rhs)
{
    GraphFillScope fill;
    if(lhs == rhs)
    {
        return;
    }
    if(lhs->extent != rhs->extent)
    {
        throw std::invalid_argument(
            "merge_axis: cannot merge axes with different extents (" +
            std::to_string(lhs->extent) + " vs " +
            std::to_string(rhs->extent) + ")");
    }

    // Union-by-size: merge the smaller membership list into the larger one.
    // Callers pass (a, b) meaning "unify these groups", not "always keep a".
    // Without this, gemm's merge_axis(fresh_activation, huge_weight_group)
    // walked every historical member every step (O(session) graph capture).
    std::shared_ptr<AxisDescriptor> *keep_slot = &lhs;
    std::shared_ptr<AxisDescriptor> *replace_slot = &rhs;
    if(lhs->members.size() < rhs->members.size())
    {
        keep_slot = &rhs;
        replace_slot = &lhs;
    }

    std::shared_ptr<AxisDescriptor> keep = *keep_slot;
    // Save the descriptor being replaced - `*replace_slot` may alias one of
    // the tensor slots we reassign inside the loop.
    std::shared_ptr<AxisDescriptor> old_desc = *replace_slot;

    if(keep->name.empty() && !old_desc->name.empty())
    {
        keep->name = old_desc->name;
    }
    if(keep->is_tiled() && old_desc->is_tiled())
    {
        if(keep->tile_sizes != old_desc->tile_sizes)
        {
            throw std::invalid_argument(
                "merge_axis: cannot merge axes with different tile sizes");
        }
    }
    else if(!keep->is_tiled() && old_desc->is_tiled())
    {
        keep->tile_sizes = old_desc->tile_sizes;
    }

    for(auto [node_ptr, axis_idx] : old_desc->members)
    {
        if(node_ptr == nullptr)
        {
            continue;
        }
        auto* node = static_cast<TensorGraph::TensorNode*>(node_ptr);
        node->mutable_axes()[axis_idx] = keep;
        keep->members.push_back({node_ptr, axis_idx});
        node->note_member_index(
            axis_idx, keep->members.size() - 1);
    }
    TensorGraph *g = nullptr;
    if(!keep->members.empty())
    {
        auto *n = static_cast<TensorGraph::TensorNode *>(
            keep->members.front().first);
        if(n != nullptr)
        {
            g = n->graph();
        }
    }
    old_desc->members.clear();
    if(g != nullptr)
    {
        g->drop_axis_group(old_desc.get());
        g->note_axis_group(keep.get());
    }
}

} // namespace nntile
