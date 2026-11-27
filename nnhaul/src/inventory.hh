/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * multi-device runtime executor's node inventory: the run-scoped,
 * static set of memory nodes with pool cardinalities and the transfer
 * channel matrix, plus the lane set derived from it.
 *
 * @file src/inventory.hh
 * The node inventory and the derived lane set with the LaneId
 * mapping. Engine configuration, never plan content: a plan references
 * NodeIds from it and carries none of it. Fixed at init; after init
 * the engine asks the device nothing.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include <nnhaul/nnhaul.hh>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace nnhaul
{

// Device nodes 0..k-1 (ascending CUDA ordinal), host node k. A small
// dense integer; range checks enforce the bound.

// A dense id over the run's lanes, static for the run.
using LaneId = std::uint16_t;

inline constexpr LaneId kNoLane = static_cast<LaneId>(0xffffu);

// ---- the env flag helper (shared by the diagnosis knobs) ----

// A VALUE-PARSED flag knob: enabled iff set to a NONZERO value, so
// `=0` is the explicit off and bare presence is not enough — the
// test scripts export the knobs as `=1` (compatible verbatim),
// and the diagnosis knobs (copy runs, the boundary/plan breakdowns)
// all parse the same way through this one helper. Sibling of
// engine.hh's env_set/env_parse_long block; lives here — the one
// header every knob-reading TU already shares (executor, planner,
// engine).
inline bool env_flag_enabled(char const *name)
{
    char const *e = std::getenv(name);
    return e != nullptr && std::atoi(e) != 0;
}

// The run-scoped, static node inventory: k device nodes
// plus one host node with their pool cardinalities, and the channel
// matrix — one bit per CONNECTED ordered node pair. From it the lane
// set is derived once: one execution lane per node plus
// one transfer lane per channel. The engine walks lanes, never
// devices.
class NodeInventory
{
  public:
    // The devices-plus-host inventory. `peer` carries the probed
    // device-device bits as a k*k row-major matrix over device NodeIds
    // (row = source i, column = destination j; the diagonal is
    // ignored). The device<->host channels — both directions, always —
    // are added by construction; a channel needs two
    // distinct endpoints, so no other pair can appear.
    // `pool_cardinalities` are the INITIAL chunks per
    // node (0 legal — preallocate nothing); `pool_limits` the soft
    // ceilings per node in chunks (0 = unlimited, the default when
    // omitted).
    NodeInventory(std::size_t devices,
        std::vector<std::size_t> const &pool_cardinalities,
        std::vector<std::uint8_t> const &peer,
        std::vector<std::size_t> const &pool_limits = {});

    std::size_t node_count() const { return n_; }          // k + 1
    std::size_t device_count() const { return n_ - 1; }    // k
    NodeId host_id() const
    {
        return static_cast<NodeId>(n_ - 1);                // k, last
    }
    bool is_device(NodeId n) const { return n + 1u < n_; }
    bool is_host(NodeId n) const { return n == host_id(); }

    // Range predicates: the node is in the inventory and the chunk is
    // in that node's pool.
    bool node_valid(NodeId n) const { return n < n_; }
    // The SOFT limit of node n's pool, in chunks
    // (0 = unlimited — growth ends at the hard limit only); fixed at
    // construction (the cap precedence: the env knob, else the
    // init argument, else unlimited).
    std::size_t pool_limit(NodeId n) const;

    // The growth act: raise node n's cardinality to
    // new_cardinality (never shrink — loud). New chunks are free by
    // construction; the binding table's grow_node carries the state.
    void grow_node(NodeId n, std::size_t new_cardinality);

    // Freeze node n's ceiling at its pool's
    // WALL: called when growth ends at an allocation error
    // below the soft limit — the budget is fixed at whatever it holds,
    // never another growth attempt. Raises only (the ceiling names
    // frozen reality, never shrinks a live pool).
    void close_growth(NodeId n, std::size_t ceiling);

    bool chunk_valid(NodeId n, std::uint32_t chunk) const;

    std::size_t pool_cardinality(NodeId n) const;

    // The channel matrix, one bit per ordered pair. A channel
    // needs two distinct endpoints; the host pairs exist by
    // construction; a device pair iff the probe said yes.
    bool has_channel(NodeId a, NodeId b) const;

    // The pure lane mapping, zero freedom. LINK and UNLINK carry
    // no lane (kNoLane); the free function over PlanOp lives in
    // plan.hh (it needs the op type).
    LaneId lane_of_exec(NodeId n) const;
    LaneId lane_of_chan(NodeId a, NodeId b) const;  // iff has_channel
    LaneId lane_count() const { return lane_count_; }

    // The swimlane rendering names lanes; the trace and
    // the census render "exec(2)" / "chan(0,3)" style names — lanes,
    // never vendors or runtimes.
    std::string lane_name(LaneId l) const;

  private:
    std::size_t n_ = 0;                       // k + 1 nodes
    std::vector<std::size_t> cardinality_;    // per node
    std::vector<std::size_t> limit_;         // per node, 0 = unlimited
    // n_ * n_ bits, row-major over NodeId: channel(a, b) = row a,
    // column b. No diagonal bit is ever set.
    std::vector<std::uint64_t> channel_;
    std::size_t channel_word_rows_ = 0;
    // The derived lane set: lanes 0..n_-1 are
    // exec(node); channel lanes follow in ascending (a, b) order of
    // the connected pairs, so the numbering is a pure function of the
    // matrix.
    std::vector<LaneId> chan_lane_;           // n_*n_, dense in (a,b)
    LaneId lane_count_ = 0;
};

} // namespace nnhaul
