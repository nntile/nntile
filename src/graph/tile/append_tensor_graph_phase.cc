/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tile/append_tensor_graph_phase.cc
 * Incremental TensorGraph phase lowering into TileGraph.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tile/append_tensor_graph_phase.hh"

#include <set>
#include <stdexcept>

#include "nntile/graph/tensor/graph.hh"
#include "nntile/graph/tile/graph.hh"
#include "nntile/graph/tile/lower_from_tensor.hh"

namespace nntile::graph
{

namespace
{

std::string tile_node_name(
    std::string const& logical_name,
    Index group_id,
    Index linear,
    Index grid_volume)
{
    std::string base =
        logical_name + "__g" + std::to_string(static_cast<long long>(group_id));
    if(grid_volume == 1)
    {
        return base;
    }
    return base + "__t" + std::to_string(static_cast<long long>(linear));
}

void collect_phase_tensors(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    std::set<TensorGraph::TensorNode const*>& out)
{
    for(TensorGraph::TensorNode const* t : phase.carried_tensors)
    {
        if(t != nullptr)
        {
            out.insert(t);
        }
    }
    const auto& ops = tg.ops();
    for(size_t i = phase.op_begin; i < phase.op_end; ++i)
    {
        for(TensorGraph::TensorNode* in : ops[i]->inputs())
        {
            out.insert(in);
        }
        for(TensorGraph::TensorNode* ot : ops[i]->outputs())
        {
            out.insert(ot);
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
    for(Index lin = 0; lin < vol; ++lin)
    {
        lay.grid_coord_from_linear(lin, grid_coord);
        const std::vector<Index> tile_shape =
            lay.tile_shape_at(grid_coord);
        const std::string tname =
            tile_node_name(tensor_node->name(), group_id, lin, vol);
        TileGraph::TileNode* tile_node_ptr = tile_graph.data(
            tile_shape,
            tname,
            tensor_node->dtype());
        if(tensor_node->is_input())
        {
            tile_node_ptr->mark_input(true);
        }
        if(tensor_node->is_output())
        {
            tile_node_ptr->mark_output(true);
        }
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
    Index group_id)
{
    std::vector<TileGraph::TileNode*> tiles = create_tile_nodes_only(
        tile_graph, tensor_node, lay, group_id);
    attach_new_descriptor(tile_graph, tensor_node, lay, tiles);
    return tiles;
}

} // namespace

void append_tensor_graph_phase(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    TensorGraphTiling const& tiling,
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

    tile_graph.set_tiling_scheme(
        std::make_shared<TensorGraphTiling>(tiling));

    std::set<TensorGraph::TensorNode const*> touched;
    collect_phase_tensors(tg, phase, touched);

    for(TensorGraph::TensorNode const* t : touched)
    {
        const TensorAxisLayout* lay = tiling.find(t);
        if(lay == nullptr)
        {
            throw std::runtime_error(
                "append_tensor_graph_phase: missing tiling for tensor '" +
                t->name() + "'");
        }
        const std::string fp = lay->layout_fingerprint();
        auto fp_it = state.tensor_layout_fp.find(t);
        const bool have_tiles =
            state.tensor_to_tiles.count(t) != 0 &&
            fp_it != state.tensor_layout_fp.end();

        if(!have_tiles)
        {
            const Index gid = state.next_tile_group_id++;
            std::vector<TileGraph::TileNode*> tiles =
                build_tile_nodes(tile_graph, t, *lay, gid);
            state.tensor_to_tiles[t] = tiles;
            state.tensor_layout_fp[t] = fp;
            tile_map[t] = tiles;
            continue;
        }

        if(fp_it->second != fp)
        {
            throw std::runtime_error(
                "append_tensor_graph_phase: tensor '" + t->name() +
                "' uses a different tiling than in an earlier phase "
                "(layout_fingerprint mismatch). Reusing the same data handle "
                "under a new tiling is not supported yet.");
        }

        tile_map[t] = state.tensor_to_tiles[t];
    }

    lower_tensor_ops_to_tile_graph(
        tg,
        tile_graph,
        tile_map,
        phase.op_begin,
        phase.op_end);
}

void compile_incremental_nn_phase(
    FinishedTensorPhase const& exec_phase,
    NNGraph& nn_graph_for_suffix,
    TensorGraphTiling const& tiling,
    TileGraph& tile_graph,
    TileGraph::Runtime& runtime,
    TileGraphIncrementalState& state,
    TensorNodeToTileMap& tile_map,
    bool archive_phase)
{
    if(exec_phase.tensor_graph == nullptr)
    {
        throw std::invalid_argument(
            "compile_incremental_nn_phase: finished phase has null tensor_graph");
    }
    std::size_t const tile_op_begin = tile_graph.num_ops();
    append_tensor_graph_phase(
        *exec_phase.tensor_graph,
        exec_phase.snapshot,
        tiling,
        tile_graph,
        state,
        tile_map);
    std::size_t const tile_op_end = tile_graph.num_ops();
    runtime.compile();
    if(archive_phase)
    {
        nn_graph_for_suffix.push_tensor_phase_archive(TensorPhaseArchiveEntry{
            exec_phase.snapshot,
            tile_op_begin,
            tile_op_end});
    }
    nn_graph_for_suffix.bump_auto_tensor_name_phase_suffix_after_compile();
    nn_graph_for_suffix.clear_pending_compile_if_same(exec_phase);
}

} // namespace nntile::graph
