/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/append_tensor_graph_phase.hh
 * Incrementally lower sealed TensorGraph phases into one TileGraph.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>
#include <nntile/defs.h>
#include <nntile/tensor/tensor_graph_phase_transform.hh>
#include <nntile/tensor/tensor_graph_tiling.hh>
#include <nntile/tile/graph_decl.hh>
#include <nntile/tile/lowering_context.hh>


namespace nntile
{

//! Mutable state for incremental tensor-to-tile lowering across phases.
//! Tile nodes use names ``logical_name__gK`` or ``logical_name__gK__tJ``.
struct TileGraphIncrementalState
{
    TensorNodeToTileMap tensor_to_tiles;
    //! Dense layout fingerprint hashes (O(1) reuse checks).
    TensorNodeIdMap<std::uint64_t> tensor_layout_fp;
    Index next_tile_group_id = 0;
};

//! Append one sealed phase: ensure tile nodes for touched tensors (reuse when
//! layout matches), then lower tensor ops in ``[phase.op_begin, phase.op_end)``.
//! Updates \p state and \p tile_map in sync.
//! Prefer the shared_ptr overload so the tiling map is not deep-copied.
void append_tensor_graph_phase(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    std::shared_ptr<TensorGraphTiling const> tiling,
    TileGraph& tile_graph,
    TileGraphIncrementalState& state,
    TensorNodeToTileMap& tile_map);

inline void append_tensor_graph_phase(
    TensorGraph const& tg,
    TensorGraph::PhaseSnapshot const& phase,
    TensorGraphTiling const& tiling,
    TileGraph& tile_graph,
    TileGraphIncrementalState& state,
    TensorNodeToTileMap& tile_map)
{
    append_tensor_graph_phase(
        tg,
        phase,
        std::make_shared<TensorGraphTiling>(tiling),
        tile_graph,
        state,
        tile_map);
}


} // namespace nntile
