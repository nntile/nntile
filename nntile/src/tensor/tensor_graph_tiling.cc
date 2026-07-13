#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/tensor_graph_tiling.cc
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/tensor_graph_tiling.hh"

#include <algorithm>
#include <sstream>

#include "nntile/tensor/axis_descriptor.hh"
#include "nntile/tensor/graph.hh"

namespace nntile
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

Index TensorAxisLayout::tile_index_containing(
    Index dim, Index global_index) const
{
    if(dim < 0 || static_cast<size_t>(dim) >= shape_.size())
    {
        throw std::out_of_range("TensorAxisLayout::tile_index_containing: dim");
    }
    if(global_index < 0 || global_index >= shape_[static_cast<size_t>(dim)])
    {
        throw std::out_of_range(
            "TensorAxisLayout::tile_index_containing: global_index");
    }
    const auto& origin = axis_origin_[static_cast<size_t>(dim)];
    auto it = std::lower_bound(origin.begin(), origin.end(), global_index + 1);
    return static_cast<Index>((it - origin.begin()) - 1);
}

std::string const &TensorAxisLayout::layout_fingerprint() const
{
    if(!fingerprint_.empty())
    {
        return fingerprint_;
    }
    std::ostringstream o;
    for(Index g : grid_shape_)
    {
        o << static_cast<long long>(g) << ',';
    }
    o << '|';
    for(const auto& seg : segments_)
    {
        for(Index s : seg)
        {
            o << static_cast<long long>(s) << ',';
        }
        o << ';';
    }
    fingerprint_ = o.str();
    return fingerprint_;
}

std::uint64_t TensorAxisLayout::layout_fingerprint_hash() const
{
    if(fingerprint_hash_ready_)
    {
        return fingerprint_hash_;
    }
    // FNV-1a 64-bit over grid shape + segment lengths (same info as string fp).
    std::uint64_t h = 14695981039346656037ull;
    auto mix = [&h](std::uint64_t x)
    {
        h ^= x;
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(grid_shape_.size()));
    for(Index g : grid_shape_)
    {
        mix(static_cast<std::uint64_t>(g));
    }
    mix(static_cast<std::uint64_t>(segments_.size()));
    for(const auto &seg : segments_)
    {
        mix(static_cast<std::uint64_t>(seg.size()));
        for(Index s : seg)
        {
            mix(static_cast<std::uint64_t>(s));
        }
    }
    fingerprint_hash_ = h;
    fingerprint_hash_ready_ = true;
    return fingerprint_hash_;
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

void TensorGraphTiling::set_layout(
    const TensorGraph::TensorNode *node, TensorAxisLayout layout)
{
    if(node == nullptr)
    {
        return;
    }
    auto const id = static_cast<size_t>(node->id());
    if(id >= layouts_by_id_.size())
    {
        layouts_by_id_.resize(id + 1);
    }
    layouts_by_id_[id] = std::move(layout);
}

TensorGraphTiling TensorGraphTiling::from_tensor_graph(const TensorGraph& tg)
{
    TensorGraphTiling out;
    for(const auto& tn : tg.tensor_nodes())
    {
        out.set_layout(tn.get(), TensorAxisLayout(tn.get()));
    }
    return out;
}

namespace
{

void collect_phase_touched(
    const TensorGraph& tg,
    const TensorGraph::PhaseSnapshot& phase,
    std::vector<const TensorGraph::TensorNode*>& touched,
    std::uint32_t gen)
{
    auto note = [&](const TensorGraph::TensorNode* t)
    {
        if(t == nullptr || t->touch_gen() == gen)
        {
            return;
        }
        t->set_touch_gen(gen);
        touched.push_back(t);
    };
    for(const TensorGraph::TensorNode* t : phase.carried_tensors)
    {
        note(t);
    }
    const auto& ops = tg.ops();
    for(size_t i = phase.op_begin; i < phase.op_end; ++i)
    {
        if(i >= ops.size() || ops[i] == nullptr)
        {
            continue;
        }
        for(TensorGraph::TensorNode* in : ops[i]->inputs())
        {
            note(in);
        }
        for(TensorGraph::TensorNode* ot : ops[i]->outputs())
        {
            note(ot);
        }
    }
}

} // namespace

TensorGraphTiling TensorGraphTiling::from_phase(
    const TensorGraph& tg,
    const TensorGraph::PhaseSnapshot& phase)
{
    TensorGraphTiling out;
    out.ensure_phase_layouts(tg, phase);
    return out;
}

void TensorGraphTiling::ensure_phase_layouts(
    const TensorGraph& tg,
    const TensorGraph::PhaseSnapshot& phase)
{
    std::vector<const TensorGraph::TensorNode*> touched;
    // Local generation counter: touch_gen is only compared within this call.
    static std::uint32_t next_gen = 1;
    std::uint32_t const gen = next_gen++;
    if(next_gen == 0)
    {
        next_gen = 1;
    }
    collect_phase_touched(tg, phase, touched, gen);
    for(const TensorGraph::TensorNode* t : touched)
    {
        if(find(t) != nullptr)
        {
            continue;
        }
        set_layout(t, TensorAxisLayout(t));
    }
}

const TensorAxisLayout* TensorGraphTiling::find(
    const TensorGraph::TensorNode* node) const
{
    if(node == nullptr)
    {
        return nullptr;
    }
    auto const id = static_cast<size_t>(node->id());
    if(id >= layouts_by_id_.size() || !layouts_by_id_[id].has_value())
    {
        return nullptr;
    }
    return &(*layouts_by_id_[id]);
}

} // namespace nntile
