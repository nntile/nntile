/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file src/planner_api.hh
 * Private glue between the engine and the public planner interface
 * <nnhaul/planner.hh>: the topology/sink adapters, the in-tree LRU path,
 * and the NNHAUL_PLANNER dynamic loader.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "lru_planner.hh"

#include <nnhaul/planner.hh>

namespace nnhaul::planner
{

//! Planner chosen for this process: nullptr = the in-tree LRU planner,
//! otherwise a plugin loaded from NNHAUL_PLANNER.
Planner *load_from_env();

//! Plan one batch through the chosen planner. Both paths emit the same
//! public vocabulary; the engine validates the result with the same
//! checker either way.
PlannedBatch run(Planner *plugin, PlanRequest const &request,
    NodeInventory const &inventory, TileTable const &tiles,
    std::vector<std::uint64_t> const &channel_rates);

} // namespace nnhaul::planner
