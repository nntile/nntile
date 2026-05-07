/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/tile/append_tensor_graph_phase.hh
 * Incrementally lower sealed TensorGraph phases into one TileGraph.
 *
 * Intended workflow (dynamic graph fill-in): record on ``NNGraph``, call
 * ``finish_phase()`` (seal + stash slice for compile), optionally edit
 * ``tensor_graph()``, then ``NNGraph::lower_and_compile`` (append + ``compile``
 * + archive + auto suffix bump).  Advanced users may call
 * ``compile_incremental_nn_phase`` or ``append_tensor_graph_phase`` +
 * ``compile()`` directly instead.
 *
 * ``TensorGraph::seal_phase()`` seeds lowering with every node marked
 * ``mark_input`` or ``mark_output``; op inputs/outputs add the rest of each
 * phase's tensors.  Those persistent buffers must keep the same
 * tiling across phases: if ``layout_fingerprint()`` differs from an earlier
 * phase, ``append_tensor_graph_phase`` throws (switching tiling on an
 * existing logical tensor will be supported later).  When the fingerprint
 * matches, existing tile nodes
 * and tensor descriptors are reused so StarPU buffers stay wired across
 * compiles.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <map>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/nn/graph.hh>
#include <nntile/graph/tensor/tensor_graph_phase_transform.hh>
#include <nntile/graph/tensor/tensor_graph_tiling.hh>
#include <nntile/graph/tile/graph_decl.hh>
#include <nntile/graph/tile/lowering_context.hh>

namespace nntile::graph
{

//! Mutable state for incremental tensor-to-tile lowering across phases.
//! Tile nodes use names ``logical_name__gK`` or ``logical_name__gK__tJ``.
struct TileGraphIncrementalState
{
    TensorNodeToTileMap tensor_to_tiles;
    std::map<TensorGraph::TensorNode const*, std::string> tensor_layout_fp;
    Index next_tile_group_id = 0;
};

//! Append one sealed phase: ensure tile nodes for touched tensors (reuse when
//! layout matches), then lower tensor ops in ``[phase.op_begin, phase.op_end)``.
//! Updates \p state and \p tile_map in sync.
void append_tensor_graph_phase(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    TensorGraphTiling const& tiling,
    TileGraph& tile_graph,
    TileGraphIncrementalState& state,
    TensorNodeToTileMap& tile_map);

//! Lower \p exec_phase into ``tile_graph``, ``compile()`` \p runtime, optionally
//! ``push_tensor_phase_archive`` on ``nn_graph_for_suffix``, then bump auto
//! module suffix tags when ``NNGraph::enable_auto_tensor_name_phase_suffix`` is
//! on.  \p tiling must describe tensors in ``*exec_phase.tensor_graph``.
void compile_incremental_nn_phase(
    FinishedTensorPhase const& exec_phase,
    NNGraph& nn_graph_for_suffix,
    TensorGraphTiling const& tiling,
    TileGraph& tile_graph,
    TileGraph::Runtime& runtime,
    TileGraphIncrementalState& state,
    TensorNodeToTileMap& tile_map,
    bool archive_phase = true);

} // namespace nntile::graph
