/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/tile/ddp.hh
 * DDP compile policy: split a named axis into worker-count tiles and
 * rewrite pending tile ops (phase-local full-sized accumulators + ADD).
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/tensor/graph_decl.hh>
#include <nntile/tile/graph_decl.hh>

#include <string>
#include <vector>

namespace nntile::tile
{

//! Even split of ``extent`` into ``n_replicas`` positive chunks.
//! Last tiles absorb the remainder; throws if any chunk would be zero.
std::vector<Index> ddp_tile_sizes(Index extent, int n_replicas);

//! Set tile sizes on every live axis group named ``axis``.
//! Replica count is ``sched::count_execution_workers()``. No-op when
//! that count is 1. Skips groups already tiled with the same sizes.
void apply_ddp_axis_tiling(
    TensorGraph &tensor_graph,
    std::string const &axis);

//! Rewrite pending tile ops ``[op_begin, op_end)``.
//! Sharded compute gets ``device_hint = replica``. Weight-grad-like
//! writes (no DDP axis on dest, batched input) are rewired to
//! phase-local full-sized tiles, then ``ADD`` into the canonical dest.
void rewrite_ddp_pending(
    TileGraph &tile_graph,
    size_t op_begin,
    size_t op_end,
    std::string const &axis);

} // namespace nntile::tile
