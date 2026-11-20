#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tile_graph/append_tensor_graph_phase.cc
 * Incremental TensorGraph phase lowering into TileGraph.
 *
 * @version 1.1.0
 * */

#include "nntile/tile/append_tensor_graph_phase.hh"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "nntile/tensor/graph.hh"
#include "nntile/tile/graph.hh"
#include <nntile/runtime.hh>
#include "nntile/tile/lower_from_tensor.hh"
#include "nntile/tensor/graph_data_node.hh"


namespace nntile
{

namespace
{

//! [C1 diag] env-gated compile sub-pass brackets (NNTILE_COMPILE_BREAKDOWN).
bool compile_breakdown_enabled()
{
    return env_flag_enabled("NNTILE_COMPILE_BREAKDOWN");
}

double breakdown_ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

void tile_node_name(
    std::string& out,
    std::string const& logical_name,
    Index group_id,
    Index linear,
    Index grid_volume)
{
    // Build into a caller-reused buffer: the old form made 3-5 string
    // temporaries per tile (the build pass's dominant allocation churn).
    out.clear();
    out += logical_name;
    out += "__g";
    out += std::to_string(static_cast<long long>(group_id));
    if(grid_volume != 1)
    {
        out += "__t";
        out += std::to_string(static_cast<long long>(linear));
    }
}

//! Dense id bitmap scratch shared by the two collect walks below (the
//! random per-node touch_gen writes were the last scatter in the
//! recorder-side compile passes).
class TensorIdBits
{
  public:
    explicit TensorIdBits(TensorGraph const& tg)
        : bits_(tg.num_data(), 0)
    {
    }
    bool note(TensorGraph::TensorNode const* t)
    {
        if(t == nullptr)
        {
            return false;
        }
        std::size_t const id = static_cast<std::size_t>(t->id());
        // id==slot invariant: TensorNode ids are dense slots into the
        // graph's data-node space (bits_ is sized by it); fail loudly on
        // a violation instead of writing past the array.
        check_tensor_id_slot(t->id(), bits_.size(),
            "append_tensor_graph_phase: TensorIdBits::note");
        if(bits_[id] != 0)
        {
            return false;
        }
        bits_[id] = 1;
        return true;
    }

  private:
    std::vector<std::uint8_t> bits_;
};

void collect_phase_tensors(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    std::vector<TensorGraph::TensorNode const*>& out,
    TensorIdBits& bits)
{
    auto note = [&](TensorGraph::TensorNode const* t)
    {
        if(bits.note(t))
        {
            out.push_back(t);
        }
    };
    for(TensorGraph::TensorNode const* t : phase.carried_tensors)
    {
        note(t);
    }
    const auto& ops = tg.ops();
    for(size_t i = phase.op_begin; i < phase.op_end; ++i)
    {
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

std::vector<TileGraph::TileNode*> create_tile_nodes_only(
    TileGraph& tile_graph,
    TensorGraph::TensorNode const* tensor_node,
    TensorAxisLayout const& lay,
    Index group_id)
{
    const Index vol = lay.grid_volume();
    std::vector<TileGraph::TileNode*> tiles;
    tiles.reserve(static_cast<size_t>(vol));
    std::vector<Index> grid_coord;
    std::string tname;
    for(Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, grid_coord);
        const std::vector<Index> tile_shape =
            lay.tile_shape_at(grid_coord);
        tile_node_name(tname, tensor_node->name(), group_id, lin, vol);
        TileGraph::TileNode* tile_node_ptr = tile_graph.data(
            tile_shape,
            tname,
            tensor_node->dtype());
        tiles.push_back(tile_node_ptr);
    }
    return tiles;
}

void attach_new_descriptor(
    TileGraph& tile_graph,
    TensorGraph::TensorNode const* tensor_node,
    TensorAxisLayout const& lay,
    std::vector<TileGraph::TileNode*>& tiles)
{
    TileGraph::TensorDescriptor desc;
    desc.tensor_name = tensor_node->name();
    desc.tensor_shape = tensor_node->shape();
    desc.tile_shape = lay.max_tile_extents();
    desc.grid_shape = lay.grid_shape();
    desc.dtype = tensor_node->dtype();
    desc.tiles = tiles;
    desc.source_node = const_cast<TensorGraph::TensorNode*>(tensor_node);

    TileGraph::TensorDescriptor* desc_ptr =
        tile_graph.add_tensor_descriptor(std::move(desc));

    std::vector<Index> grid_coord;
    const Index vol = lay.grid_volume();
    for(Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, grid_coord);
        tiles[static_cast<size_t>(lin)]->set_tensor_info(
            desc_ptr, grid_coord);
    }
}

std::vector<TileGraph::TileNode*> build_tile_nodes(
    TileGraph& tile_graph,
    TensorGraph::TensorNode const* tensor_node,
    TensorAxisLayout const& lay,
    Index group_id,
    double* attach_ms_acc)
{
    std::vector<TileGraph::TileNode*> tiles = create_tile_nodes_only(
        tile_graph, tensor_node, lay, group_id);
    if(attach_ms_acc == nullptr)
    {
        attach_new_descriptor(tile_graph, tensor_node, lay, tiles);
        return tiles;
    }
    // Diag-only bracket (NNTILE_COMPILE_BREAKDOWN); no clock reads when
    // the env flag is off.
    auto const ta0 = std::chrono::steady_clock::now();
    attach_new_descriptor(tile_graph, tensor_node, lay, tiles);
    *attach_ms_acc += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ta0).count();
    return tiles;
}

} // namespace

void append_tensor_graph_phase(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    std::shared_ptr<TensorGraphTiling const> tiling,
    TileGraph& tile_graph,
    TileGraphIncrementalState& state,
    TensorNodeToTileMap& tile_map)
{
    if(phase.empty())
    {
        throw std::invalid_argument(
            "append_tensor_graph_phase: empty phase snapshot");
    }
    if(phase.op_end > tg.ops().size())
    {
        throw std::out_of_range(
            "append_tensor_graph_phase: phase.op_end > num_ops");
    }
    if(tiling == nullptr)
    {
        throw std::invalid_argument(
            "append_tensor_graph_phase: tiling must be non-null");
    }

    tile_graph.set_tiling_scheme(tiling);

    bool const cbrk = compile_breakdown_enabled();
    auto const t_collect = std::chrono::steady_clock::now();
    std::vector<TensorGraph::TensorNode const*> touched;
    touched.reserve(
        (phase.op_end - phase.op_begin) + phase.carried_tensors.size()
        + 8);
    TensorIdBits bits(tg);
    collect_phase_tensors(tg, phase, touched, bits);
    double const collect_ms =
        cbrk ? breakdown_ms_since(t_collect) : 0.0;
    auto const t_build = std::chrono::steady_clock::now();
    double create_ms = 0.0;
    double attach_ms = 0.0;
    double maps_ms = 0.0;
    std::size_t new_tensors = 0;

    for(TensorGraph::TensorNode const* t : touched)
    {
        const TensorAxisLayout* lay = tiling->find(t);
        if(lay == nullptr)
        {
            throw std::runtime_error(
                "append_tensor_graph_phase: missing tiling for tensor '" +
                t->name() + "'");
        }
        std::uint64_t const fp = lay->layout_fingerprint_hash();
        std::uint64_t const *fp_ptr = state.tensor_layout_fp.try_get(t);
        const bool have_tiles =
            state.tensor_to_tiles.contains(t) && fp_ptr != nullptr;

        if(!have_tiles)
        {
            const Index gid = state.next_tile_group_id++;
            // Diag-only brackets (NNTILE_COMPILE_BREAKDOWN): no clock
            // reads on the ungated path.
            std::chrono::steady_clock::time_point tc0{};
            if(cbrk)
            {
                tc0 = std::chrono::steady_clock::now();
            }
            double *attach_ms_this = nullptr;
            double attach_before = 0.0;
            if(cbrk)
            {
                attach_ms_this = &attach_ms;
                attach_before = attach_ms;
            }
            std::vector<TileGraph::TileNode*> tiles =
                build_tile_nodes(tile_graph, t, *lay, gid,
                    attach_ms_this);
            std::chrono::steady_clock::time_point tc1{};
            if(cbrk)
            {
                tc1 = std::chrono::steady_clock::now();
            }
            state.tensor_to_tiles[t] = tiles;
            state.tensor_layout_fp[t] = fp;
            tile_map[t] = std::move(tiles);
            if(cbrk)
            {
                // create_ms EXCLUDES attach time (reported separately
                // as attach_ms) so the three build2 fields are disjoint.
                double const build_wall_ms =
                    std::chrono::duration<double, std::milli>(
                        tc1 - tc0).count();
                create_ms += build_wall_ms
                    - (attach_ms - attach_before);
                maps_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tc1).count();
            }
            ++new_tensors;
            continue;
        }

        if(*fp_ptr != fp)
        {
            throw std::runtime_error(
                "append_tensor_graph_phase: tensor '" + t->name() +
                "' uses a different tiling than in an earlier phase "
                "(layout_fingerprint mismatch). Reusing the same data handle "
                "under a new tiling is not supported yet.");
        }

        tile_map[t] = state.tensor_to_tiles[t];
    }
    // Snapshot build_ms BEFORE the build2 print: the stderr write must
    // not pollute the enclosing phase bracket.
    double const build_ms = cbrk ? breakdown_ms_since(t_build) : 0.0;
    if(cbrk)
    {
        std::fprintf(stderr,
            "[nntile-cbrk] build2 create_ms=%.3f attach_ms=%.3f "
            "maps_ms=%.3f\n",
            create_ms, attach_ms, maps_ms);
    }
    auto const t_lower = std::chrono::steady_clock::now();

    lower_tensor_ops_to_tile_graph(
        tg,
        tile_graph,
        tile_map,
        phase.op_begin,
        phase.op_end);
    if(cbrk)
    {
        std::fprintf(stderr,
            "[nntile-cbrk] phase phase_ops=%zu collect_ms=%.3f "
            "build_ms=%.3f lower_ms=%.3f touched=%zu new_tensors=%zu\n",
            static_cast<std::size_t>(phase.op_end - phase.op_begin),
            collect_ms, build_ms, breakdown_ms_since(t_lower),
            touched.size(), new_tensors);
    }
}


} // namespace nntile
