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
#include "nntile/tensor/ops/logsumexp.hh"
#include "nntile/tensor/ops/subtract_indexed_outputs.hh"
#include "nntile/tensor/ops/total_sum_accum.hh"
#include "nntile/tensor/graph_data_node.hh"


namespace nntile
{

namespace
{

//! Constrain axis tiling around ops whose tile kernels cannot honor an
//! arbitrary segmentation. The kernels behind LOGSUMEXP,
//! TOTAL_SUM_ACCUM and SUBTRACT_INDEXED_OUTPUTS index their pair /
//! class axis with global positions against a local tile extent (and
//! LOGSUMEXP additionally requires the trailing maxsumexp pair dim in
//! one tile), so a tiled policy on those axes mis-indexes memory or
//! fails the lowering's src/dst grid-volume check. The axes in
//! question are per-op details (a maxsumexp pair axis never merges
//! with the logits class axis), so dropping their tiling loses no
//! requested layout on user-named axes.
void constrain_reduction_tiles(const TensorGraph &tg)
{
    for (const auto &op : tg.ops())
    {
        if (op == nullptr)
        {
            continue;
        }
        if (!op->constrains_reduction_tiles())
        {
            continue;
        }
        if (auto const *lse =
                dynamic_cast<tensor::TensorLogsumexpOp const *>(op.get()))
        {
            if (lse->src != nullptr && lse->src->ndim() > 0)
            {
                lse->src->axis(lse->src->ndim() - 1)->clear_tiling();
            }
        }
        else if (auto const *tsa = dynamic_cast<
                     tensor::TensorTotalSumAccumOp const *>(op.get()))
        {
            if (tsa->src != nullptr && tsa->src->ndim() > 0)
            {
                tsa->src->axis(tsa->src->ndim() - 1)->clear_tiling();
            }
        }
        else if (auto const *sio = dynamic_cast<
                     tensor::TensorSubtractIndexedOutputsOp const *>(
                     op.get()))
        {
            if (sio->dst != nullptr && sio->dst->ndim() > 0)
            {
                sio->dst->axis(sio->dst->ndim() - 1)->clear_tiling();
            }
        }
    }
}

} // namespace


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
    grid_shape_.assign(static_cast<size_t>(ndim), 1);
    grid_volume_ = 1;

    // Pass 1: grid shape per axis (segment count) + validation.
    for(Index d = 0; d < ndim; ++d)
    {
        const AxisDescriptor* ax = axes[static_cast<size_t>(d)].get();
        const std::vector<Index>* seg;
        std::vector<Index> one;
        if(!ax->is_tiled())
        {
            one = {shape_[static_cast<size_t>(d)]};
            seg = &one;
        }
        else
        {
            seg = &ax->tile_sizes;
        }
        Index sum = 0;
        for(Index svv : *seg)
        {
            if(svv <= 0)
            {
                throw std::invalid_argument(
                    "TensorAxisLayout: non-positive segment on axis " +
                    std::to_string(d) + " for '" + node->name() + "'");
            }
            sum += svv;
        }
        if(sum != shape_[static_cast<size_t>(d)])
        {
            throw std::invalid_argument(
                "TensorAxisLayout: segment sum != extent on axis " +
                std::to_string(d) + " for '" + node->name() + "'");
        }
        grid_shape_[static_cast<size_t>(d)] = static_cast<Index>(seg->size());
        grid_volume_ *= grid_shape_[static_cast<size_t>(d)];
    }

    // Pass 2: flat fill (one buffer for segments with +1 padding per
    // axis, one for prefix origins, one row table shared by both).
    row_.assign(static_cast<size_t>(ndim) + 1, 0);
    std::size_t total = 0;
    for(Index d = 0; d < ndim; ++d)
    {
        row_[static_cast<std::size_t>(d) + 1] =
            row_[static_cast<std::size_t>(d)]
            + static_cast<Index>(grid_shape_[static_cast<std::size_t>(d)]) + 1;
        total = static_cast<std::size_t>(row_[static_cast<std::size_t>(d) + 1]);
    }
    segs_.assign(total, 0);
    origins_.assign(total, 0);
    for(Index d = 0; d < ndim; ++d)
    {
        const AxisDescriptor* ax = axes[static_cast<size_t>(d)].get();
        const std::size_t base =
            static_cast<std::size_t>(row_[static_cast<std::size_t>(d)]);
        const std::size_t nseg =
            static_cast<std::size_t>(grid_shape_[static_cast<std::size_t>(d)]);
        if(!ax->is_tiled())
        {
            segs_[base] = shape_[static_cast<std::size_t>(d)];
        }
        else
        {
            std::copy(ax->tile_sizes.begin(), ax->tile_sizes.end(),
                segs_.begin() + static_cast<std::ptrdiff_t>(base));
        }
        for(std::size_t k = 0; k < nseg; ++k)
        {
            origins_[base + k + 1] =
                origins_[base + k] + segs_[base + k];
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
        ts[d] = segs_[static_cast<size_t>(row_[d])
                  + static_cast<size_t>(grid_coord[d])];
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
        global_out[d] = origins_[static_cast<size_t>(row_[d])
                        + static_cast<size_t>(seg_idx)]
                      + local_within_tile[d];
    }
}

std::vector<Index> TensorAxisLayout::max_tile_extents() const
{
    std::vector<Index> m(shape_.size(), 1);
    for(size_t d = 0; d + 1 < row_.size(); ++d)
    {
        const std::size_t base = static_cast<std::size_t>(row_[d]);
        const std::size_t nseg =
            static_cast<std::size_t>(grid_shape_[d]);
        for(std::size_t k = 0; k < nseg; ++k)
        {
            m[d] = std::max(m[d], segs_[base + k]);
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
    const std::size_t base = static_cast<std::size_t>(row_[static_cast<std::size_t>(dim)]);
    const std::size_t nseg =
        static_cast<std::size_t>(grid_shape_[static_cast<std::size_t>(dim)]);
    auto it = std::lower_bound(origins_.begin() + static_cast<std::ptrdiff_t>(base),
        origins_.begin() + static_cast<std::ptrdiff_t>(base + nseg + 1),
        global_index + 1);
    return static_cast<Index>((it - (origins_.begin() + static_cast<std::ptrdiff_t>(base))) - 1);
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
    for(size_t d = 0; d + 1 < row_.size(); ++d)
    {
        const std::size_t base = static_cast<std::size_t>(row_[d]);
        const std::size_t nseg =
            static_cast<std::size_t>(grid_shape_[d]);
        for(std::size_t k = 0; k < nseg; ++k)
        {
            o << static_cast<long long>(segs_[base + k]) << ',';
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
    mix(static_cast<std::uint64_t>(grid_shape_.size()));
    for(size_t d = 0; d + 1 < row_.size(); ++d)
    {
        const std::size_t base = static_cast<std::size_t>(row_[d]);
        const std::size_t nseg =
            static_cast<std::size_t>(grid_shape_[d]);
        mix(static_cast<std::uint64_t>(nseg));
        for(std::size_t k = 0; k < nseg; ++k)
        {
            mix(static_cast<std::uint64_t>(segs_[base + k]));
        }
    }
    fingerprint_hash_ = h;
    fingerprint_hash_ready_ = true;
    return fingerprint_hash_;
}

void TensorAxisLayout::tile_axis_segment_range(
    Index dim,
    Index seg,
    Index& global_lo,
    Index& global_hi_inclusive) const
{
    if(dim < 0 || static_cast<size_t>(dim) >= grid_shape_.size())
    {
        throw std::out_of_range(
            "TensorAxisLayout::tile_axis_segment_range: dim");
    }
    if(seg < 0 || seg >= grid_shape_[static_cast<size_t>(dim)])
    {
        throw std::out_of_range(
            "TensorAxisLayout::tile_axis_segment_range: seg");
    }
    const std::size_t pos =
        static_cast<std::size_t>(row_[static_cast<std::size_t>(dim)])
        + static_cast<std::size_t>(seg);
    global_lo = origins_[pos];
    global_hi_inclusive = global_lo + segs_[pos] - 1;
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
    const std::size_t pos =
        static_cast<std::size_t>(row_[static_cast<std::size_t>(dim)])
        + static_cast<std::size_t>(seg);
    global_lo = origins_[pos];
    global_hi_inclusive = global_lo + segs_[pos] - 1;
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
    constrain_reduction_tiles(tg);
    TensorGraphTiling out;
    for(const auto& tn : tg.tensor_nodes())
    {
        if(!tn)
        {
            continue;
        }
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
    std::vector<std::uint8_t>& bits)
{
    auto note = [&](const TensorGraph::TensorNode* t)
    {
        if(t == nullptr)
        {
            return;
        }
        std::size_t const id = static_cast<std::size_t>(t->id());
        // id==slot invariant: ids are dense slots into the graph's
        // data-node space (bits is sized by it). An out-of-range id is
        // graph corruption - fail loudly instead of silently dropping
        // the node.
        check_tensor_id_slot(t->id(), bits.size(),
            "TensorGraphTiling: collect_phase_touched");
        if(bits[id] != 0)
        {
            return;
        }
        bits[id] = 1;
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
    // ensure_phase_layouts below runs constrain_reduction_tiles.
    TensorGraphTiling out;
    out.ensure_phase_layouts(tg, phase);
    return out;
}

void TensorGraphTiling::ensure_phase_layouts(
    const TensorGraph& tg,
    const TensorGraph::PhaseSnapshot& phase)
{
    // [C5 diag] NNTILE_COMPILE_BREAKDOWN splits this pass; all clock
    // reads are gated on it (none on the ungated path).
    static bool const cbrk2 = env_flag_enabled("NNTILE_COMPILE_BREAKDOWN");
    std::chrono::steady_clock::time_point t0{};
    std::chrono::steady_clock::time_point t1{};
    std::chrono::steady_clock::time_point t2{};
    std::chrono::steady_clock::time_point t3{};
    if(cbrk2)
    {
        t0 = std::chrono::steady_clock::now();
    }
    constrain_reduction_tiles(tg);
    if(cbrk2)
    {
        t1 = std::chrono::steady_clock::now();
    }
    std::vector<const TensorGraph::TensorNode*> touched;
    std::vector<std::uint8_t> bits(tg.num_data(), 0);
    collect_phase_touched(tg, phase, touched, bits);
    if(cbrk2)
    {
        t2 = std::chrono::steady_clock::now();
    }
    std::size_t made = 0;
    for(const TensorGraph::TensorNode* t : touched)
    {
        if(find(t) != nullptr)
        {
            continue;
        }
        set_layout(t, TensorAxisLayout(t));
        ++made;
    }
    if(cbrk2)
    {
        t3 = std::chrono::steady_clock::now();
        std::fprintf(stderr,
            "[nntile-cbrk] tiling2 constrain_ms=%.3f collect_ms=%.3f "
            "setlayout_ms=%.3f made=%zu\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count(),
            std::chrono::duration<double, std::milli>(t3 - t2).count(),
            made);
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
