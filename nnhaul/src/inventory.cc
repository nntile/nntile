/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * node inventory's one act: assembling it loudly from the probed
 * facts and deriving the lane set once, as a pure function of the
 * channel matrix.
 *
 * @file src/inventory.cc
 * The construction behind src/inventory.hh: the loud constructor
 * validation, the k+1 nodes with their pool cardinalities, the channel
 * matrix — the 2k device<->host channels added by construction, the
 * probed device pairs recorded faithfully per direction, the diagonal
 * never set — and the lane set derived once at run start: exec lanes
 * 0..k, then one transfer lane per connected ordered pair in ascending
 * (a, b) order.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "inventory.hh"

#include <stdexcept>
#include <string>

namespace nnhaul
{

NodeInventory::NodeInventory(std::size_t devices,
    std::vector<std::size_t> const &pool_cardinalities,
    std::vector<std::uint8_t> const &peer,
    std::vector<std::size_t> const &pool_limits)
{
    // Loud construction: every bad input is a precise throw.
    if (devices == 0)
    {
        throw std::runtime_error(
            "node inventory: devices == 0, but the inventory always has "
            "at least one device node beside the host ");
    }
    // The id scheme: device NodeIds 0..k-1 and the
    // host NodeId k must all be representable in the uint8_t NodeId, so
    // k <= 255; a larger inventory would truncate ids silently.
    if (devices > 255u)
    {
        throw std::runtime_error("node inventory: devices = " +
            std::to_string(devices) + ", but the host NodeId k must fit "
            "the 8-bit NodeId, so at most 255 device nodes ");
    }

    n_ = devices + 1;

    if (pool_cardinalities.size() != n_)
    {
        throw std::runtime_error("node inventory: pool_cardinalities has " +
            std::to_string(pool_cardinalities.size()) +
            " entries, but the inventory has k + 1 = " + std::to_string(n_) +
            " nodes ");
    }
    // A pool may start EMPTY (cardinality 0: preallocate nothing
    // unless asked). The cardinalities here are the INITIAL ones,
    // grown later through grow_node (never shrunk), each capped by its
    // limit when one is named.
    cardinality_ = pool_cardinalities;
    limit_ = pool_limits;
    if (limit_.size() != n_)
    {
        // Default construction path: no limits named anywhere.
        limit_.assign(n_, 0);
    }
    for (std::size_t n = 0; n < n_; ++n)
    {
        if (limit_[n] != 0 && cardinality_[n] > limit_[n])
        {
            throw std::runtime_error(
                "node inventory: initial pool cardinality of node " +
                std::to_string(n) + " (" +
                std::to_string(cardinality_[n]) +
                ") exceeds its limit (" + std::to_string(limit_[n]) +
                ")");
        }
    }

    if (peer.size() != devices * devices)
    {
        throw std::runtime_error("node inventory: peer matrix has " +
            std::to_string(peer.size()) + " bytes, but a k*k matrix over "
            "k = " + std::to_string(devices) + " devices needs exactly " +
            std::to_string(devices * devices) + " ");
    }

    // The lane count must live in the dense 16-bit LaneId and leave
    // kNoLane (0xffff) reserved for "no lane" : (k + 1)
    // exec lanes plus 2k host channels plus the probed device pairs.
    {
        std::size_t channels = 2 * devices;
        for (std::size_t i = 0; i < devices; ++i)
        {
            for (std::size_t j = 0; j < devices; ++j)
            {
                if (i != j && peer[i * devices + j] != 0)
                {
                    ++channels;
                }
            }
        }
        if (n_ + channels > static_cast<std::size_t>(kNoLane))
        {
            throw std::runtime_error("node inventory: lane count " +
                std::to_string(n_ + channels) + " exceeds the 16-bit "
                "LaneId range ");
        }
    }

    // (cardinality_ and limit_ set above)

    // The channel matrix as bits, row-major over NodeId: channel(a, b)
    // = row a, column b. channel_word_rows_ is the 64-bit word count
    // of one row.
    channel_word_rows_ = (n_ + 63u) / 64u;
    channel_.assign(channel_word_rows_ * n_, 0);
    auto set_channel = [this](std::size_t a, std::size_t b) {
        channel_[channel_word_rows_ * a + (b / 64u)] |=
            (std::uint64_t{1} << (b % 64u));
    };

    // Device <-> host, both directions, ALWAYS — the 2k channels that
    // no probe gates, every device's host endpoints included. And the
    // probed device pairs, faithful per direction (a pair may be
    // unidirectional, never mirrored); the diagonal is ignored — a
    // channel needs two distinct endpoints.
    for (std::size_t i = 0; i < devices; ++i)
    {
        set_channel(i, n_ - 1);
        set_channel(n_ - 1, i);
        for (std::size_t j = 0; j < devices; ++j)
        {
            if (i != j && peer[i * devices + j] != 0)
            {
                set_channel(i, j);
            }
        }
    }

    // The lane derivation, once: lanes 0..n_-1 are
    // exec(node) — the host lane LAST among exec lanes — then one
    // transfer lane per connected ordered pair, numbered in ascending
    // (a, b) order, so the numbering is a pure function of the matrix.
    chan_lane_.assign(n_ * n_, kNoLane);
    LaneId next_lane = static_cast<LaneId>(n_);
    for (std::size_t a = 0; a < n_; ++a)
    {
        for (std::size_t b = 0; b < n_; ++b)
        {
            if (a != b &&
                ((channel_[channel_word_rows_ * a + (b / 64u)] >>
                     (b % 64u)) &
                    1u) != 0u)
            {
                chan_lane_[a * n_ + b] = next_lane;
                ++next_lane;
            }
        }
    }
    lane_count_ = next_lane;
}

bool NodeInventory::chunk_valid(NodeId n, std::uint32_t chunk) const
{
    // A pure range predicate: the node is in the inventory and the
    // chunk is in that node's pool.
    return n < n_ && static_cast<std::size_t>(chunk) < cardinality_[n];
}

std::size_t NodeInventory::pool_cardinality(NodeId n) const
{
    if (n >= n_)
    {
        throw std::runtime_error("node inventory: pool_cardinality of "
            "node " + std::to_string(static_cast<unsigned>(n)) + ", not "
            "one of the " + std::to_string(n_) + " inventory nodes");
    }
    return cardinality_[n];
}

std::size_t NodeInventory::pool_limit(NodeId n) const
{
    if (n >= n_)
    {
        throw std::runtime_error("node inventory: pool_limit of node " +
            std::to_string(n) + " outside the inventory (" +
            std::to_string(n_) + " nodes)");
    }
    return limit_[n];
}

void NodeInventory::close_growth(NodeId n, std::size_t ceiling)
{
    // The pool's growth ended at a WALL below its soft limit (the
    // first allocation error): the pool's budget is FIXED at whatever
    // it holds, and the inventory's ceiling becomes that cardinality —
    // the planner's growth asks stop here. Never shrinks a pool (the
    // ceiling names frozen reality).
    if (n >= n_)
    {
        throw std::runtime_error("node inventory: close_growth of node " +
            std::to_string(n) + " outside the inventory (" +
            std::to_string(n_) + " nodes)");
    }
    if (ceiling < cardinality_[n])
    {
        ceiling = cardinality_[n];
    }
    limit_[n] = ceiling;
}

void NodeInventory::grow_node(NodeId n, std::size_t new_cardinality)
{
    if (n >= n_)
    {
        throw std::runtime_error("node inventory: grow_node of node " +
            std::to_string(n) + " outside the inventory (" +
            std::to_string(n_) + " nodes)");
    }
    if (new_cardinality < cardinality_[n])
    {
        throw std::runtime_error(
            "node inventory: grow_node(" + std::to_string(n) + ", " +
            std::to_string(new_cardinality) + ") would shrink node " +
            std::to_string(n) + "'s pool of " +
            std::to_string(cardinality_[n]) + " chunks — refused");
    }
    if (limit_[n] != 0 && new_cardinality > limit_[n])
    {
        throw std::runtime_error(
            "node inventory: grow_node(" + std::to_string(n) + ", " +
            std::to_string(new_cardinality) + ") exceeds node " +
            std::to_string(n) + "'s soft limit of " +
            std::to_string(limit_[n]) + " chunks — refused");
    }
    cardinality_[n] = new_cardinality;
}

bool NodeInventory::has_channel(NodeId a, NodeId b) const
{
    // A channel needs two distinct endpoints; ids outside the
    // inventory name no endpoint. Otherwise the matrix bit.
    if (a == b || a >= n_ || b >= n_)
    {
        return false;
    }
    return ((channel_[channel_word_rows_ * a + (b / 64u)] >> (b % 64u)) &
        1u) != 0u;
}

LaneId NodeInventory::lane_of_exec(NodeId n) const
{
    // TILE_OP -> exec(node); the exec lanes are numbered
    // by NodeId, so the mapping is the identity. A node outside the
    // inventory has no lane — loud, never a wrong id.
    if (n >= n_)
    {
        throw std::runtime_error("node inventory: lane_of_exec of node " +
            std::to_string(static_cast<unsigned>(n)) + ", not one of the "
            + std::to_string(n_) + " inventory nodes");
    }
    return static_cast<LaneId>(n);
}

LaneId NodeInventory::lane_of_chan(NodeId a, NodeId b) const
{
    // COPY -> chan(src.node, dst.node) iff the channel
    // exists; otherwise no lane (the planner makes a staging hop
    // explicitly).
    if (!has_channel(a, b))
    {
        return kNoLane;
    }
    return chan_lane_[static_cast<std::size_t>(a) * n_ + b];
}

std::string NodeInventory::lane_name(LaneId l) const
{
    // Lanes, never vendors or runtimes: the two
    // families — exec(node) and chan(a, b) — cover the whole set.
    if (l >= lane_count_)
    {
        throw std::runtime_error("node inventory: lane_name of lane " +
            std::to_string(l) + ", outside the run's " +
            std::to_string(lane_count_) + " lanes");
    }
    if (l < n_)
    {
        return "exec(" + std::to_string(static_cast<unsigned>(l)) + ")";
    }
    for (std::size_t a = 0; a < n_; ++a)
    {
        for (std::size_t b = 0; b < n_; ++b)
        {
            if (a != b && chan_lane_[a * n_ + b] == l)
            {
                return "chan(" + std::to_string(a) + "," +
                    std::to_string(b) + ")";
            }
        }
    }
    // Unreachable: every lane below lane_count_ is one of the two
    // families by construction. Loud if ever reached anyway.
    throw std::runtime_error("node inventory: lane " + std::to_string(l) +
        " is neither an execution nor a transfer lane");
}

} // namespace nnhaul
