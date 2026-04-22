/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/tensor_graph_tiling.cc
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/tensor_graph_tiling.hh"

#include <algorithm>
#include <numeric>

#include "nntile/graph/tensor/axis_descriptor.hh"
#include "nntile/graph/tensor/graph.hh"

namespace nntile::graph
{

TensorAxisLayout::TensorAxisLayout(const TensorGraph::TensorNode* node)
{
    shape_ = node->shape();
    const Index ndim = static_cast<Index>(shape_.size());
    const auto& axes = node->axes();
    if(static_cast<size_t>(ndim) != axes.size())
    {
        throw std::runtime_error(
            "TensorAxisLayout: axes/shape mismatch for '" + node->name() + "'");
    }
    segments_.resize(static_cast<size_t>(ndim));
    axis_origin_.resize(static_cast<size_t>(ndim));
    grid_shape_.assign(static_cast<size_t>(ndim), 1);
    grid_volume_ = 1;

    for(Index d = 0; d < ndim; ++d)
    {
        const AxisDescriptor* ax = axes[static_cast<size_t>(d)].get();
        if(!ax->is_tiled())
        {
            segments_[static_cast<size_t>(d)] = {shape_[static_cast<size_t>(d)]};
        }
        else
        {
            segments_[static_cast<size_t>(d)] = ax->tile_sizes;
        }
        const auto& seg = segments_[static_cast<size_t>(d)];
        Index sum = 0;
        for(Index s : seg)
        {
            if(s <= 0)
            {
                throw std::invalid_argument(
                    "TensorAxisLayout: non-positive segment on axis " +
                    std::to_string(d) + " for '" + node->name() + "'");
            }
            sum += s;
        }
        if(sum != shape_[static_cast<size_t>(d)])
        {
            throw std::invalid_argument(
                "TensorAxisLayout: segment sum != extent on axis " +
                std::to_string(d) + " for '" + node->name() + "'");
        }
        grid_shape_[static_cast<size_t>(d)] = static_cast<Index>(seg.size());
        grid_volume_ *= grid_shape_[static_cast<size_t>(d)];

        std::vector<Index> origin(seg.size() + 1, 0);
        for(size_t k = 0; k < seg.size(); ++k)
        {
            origin[k + 1] = origin[k] + seg[k];
        }
        axis_origin_[static_cast<size_t>(d)] = std::move(origin);
    }

    dense_stride_.assign(static_cast<size_t>(ndim), 1);
    if(ndim > 0)
    {
        for(Index i = ndim - 2; i >= 0; --i)
        {
            dense_stride_[static_cast<size_t>(i)] =
                dense_stride_[static_cast<size_t>(i + 1)] *
                shape_[static_cast<size_t>(i + 1)];
        }
    }
}

Index TensorAxisLayout::grid_linear(const std::vector<Index>& grid_coord) const
{
    if(grid_coord.size() != grid_shape_.size())
    {
        throw std::invalid_argument("TensorAxisLayout::grid_linear: bad coord");
    }
    Index lin = 0;
    for(size_t d = 0; d < grid_shape_.size(); ++d)
    {
        if(grid_coord[d] < 0 || grid_coord[d] >= grid_shape_[d])
        {
            throw std::out_of_range("TensorAxisLayout::grid_linear: coord OOB");
        }
        lin = lin * grid_shape_[d] + grid_coord[d];
    }
    return lin;
}

void TensorAxisLayout::grid_coord_from_linear(
    Index linear, std::vector<Index>& grid_coord) const
{
    if(linear < 0 || linear >= grid_volume_)
    {
        throw std::out_of_range(
            "TensorAxisLayout::grid_coord_from_linear: linear out of range");
    }
    grid_coord.resize(grid_shape_.size());
    Index rem = linear;
    for(size_t d = 0; d < grid_shape_.size(); ++d)
    {
        Index stride = 1;
        for(size_t k = d + 1; k < grid_shape_.size(); ++k)
        {
            stride *= grid_shape_[k];
        }
        grid_coord[d] = rem / stride;
        rem %= stride;
    }
}

std::vector<Index> TensorAxisLayout::tile_shape_at(
    const std::vector<Index>& grid_coord) const
{
    if(grid_coord.size() != grid_shape_.size())
    {
        throw std::invalid_argument(
            "TensorAxisLayout::tile_shape_at: bad coord size");
    }
    std::vector<Index> ts(grid_shape_.size());
    for(size_t d = 0; d < grid_shape_.size(); ++d)
    {
        if(grid_coord[d] < 0 || grid_coord[d] >= grid_shape_[d])
        {
            throw std::out_of_range("TensorAxisLayout::tile_shape_at: OOB");
        }
        ts[d] = segments_[d][static_cast<size_t>(grid_coord[d])];
    }
    return ts;
}

Index TensorAxisLayout::tile_nelems_at(
    const std::vector<Index>& grid_coord) const
{
    Index n = 1;
    for(Index v : tile_shape_at(grid_coord))
    {
        n *= v;
    }
    return n;
}

void TensorAxisLayout::global_coord(
    const std::vector<Index>& grid_coord,
    const std::vector<Index>& local_within_tile,
    std::vector<Index>& global_out) const
{
    const std::vector<Index> ts = tile_shape_at(grid_coord);
    if(local_within_tile.size() != ts.size())
    {
        throw std::invalid_argument("TensorAxisLayout::global_coord: bad local");
    }
    global_out.resize(ts.size());
    for(size_t d = 0; d < ts.size(); ++d)
    {
        if(local_within_tile[d] < 0 || local_within_tile[d] >= ts[d])
        {
            throw std::out_of_range("TensorAxisLayout::global_coord: local OOB");
        }
        const Index seg_idx = grid_coord[d];
        global_out[d] = axis_origin_[d][static_cast<size_t>(seg_idx)] +
                      local_within_tile[d];
    }
}

std::vector<Index> TensorAxisLayout::max_tile_extents() const
{
    std::vector<Index> m(shape_.size(), 1);
    for(size_t d = 0; d < segments_.size(); ++d)
    {
        for(Index s : segments_[d])
        {
            m[d] = std::max(m[d], s);
        }
    }
    return m;
}

void TensorAxisLayout::tile_axis_global_range(
    const std::vector<Index>& grid_coord,
    Index dim,
    Index& global_lo,
    Index& global_hi_inclusive) const
{
    if(dim < 0 || static_cast<size_t>(dim) >= grid_shape_.size())
    {
        throw std::out_of_range("TensorAxisLayout::tile_axis_global_range: dim");
    }
    const Index seg = grid_coord[static_cast<size_t>(dim)];
    global_lo = axis_origin_[static_cast<size_t>(dim)][static_cast<size_t>(seg)];
    global_hi_inclusive =
        global_lo + segments_[static_cast<size_t>(dim)][static_cast<size_t>(seg)] -
        1;
}

Index TensorAxisLayout::dense_linear_index(
    const std::vector<Index>& global_coord) const
{
    if(global_coord.size() != shape_.size())
    {
        throw std::invalid_argument(
            "TensorAxisLayout::dense_linear_index: bad coord");
    }
    Index idx = 0;
    for(size_t d = 0; d < shape_.size(); ++d)
    {
        if(global_coord[d] < 0 || global_coord[d] >= shape_[d])
        {
            throw std::out_of_range(
                "TensorAxisLayout::dense_linear_index: global OOB");
        }
        idx += global_coord[d] * dense_stride_[d];
    }
    return idx;
}

TensorGraphTiling TensorGraphTiling::from_tensor_graph(const TensorGraph& tg)
{
    TensorGraphTiling out;
    for(const auto& tn : tg.tensor_nodes())
    {
        out.layouts_.emplace(tn.get(), TensorAxisLayout(tn.get()));
    }
    return out;
}

const TensorAxisLayout* TensorGraphTiling::find(
    const TensorGraph::TensorNode* node) const
{
    auto it = layouts_.find(node);
    return it == layouts_.end() ? nullptr : &it->second;
}

} // namespace nntile::graph
