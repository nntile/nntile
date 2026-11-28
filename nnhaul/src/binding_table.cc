/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * one binding table's acts: the free list's mechanism, the fold's O(1)
 * transitions with the reverse map kept exact, the state read's
 * queries, and the boundary halves over all nodes' segments.
 *
 * @file src/binding_table.cc
 * The acts behind src/binding_table.hh's table: FreeList's
 * seed/pop/push/remove/member, init's all-free N segments,
 * link/unlink/actualize/outdate_replicas — the complete flag-transition
 * list, nothing else flips a flag — the queries of the state read,
 * equals/snapshot as the boundary halves across ALL nodes' segments,
 * and the census render. The header states the checked/unchecked
 * contract; every loud throw below is either programmer range misuse
 * or the FreeList's own discipline, never a fold fault.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "binding_table.hh"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace nnhaul
{

// ---- the free list's mechanism  ----

void FreeList::grow(std::size_t old_count, std::size_t new_count)
{
    // Append the ids [old_count, new_count) as free
    // members, descending (the back stays the next pop, so growth
    // hands out the NEW chunks first — the fresh capacity leads the
    // free order; the seed's own fresh build differs only in laying
    // them out ascending).
    ids.reserve(ids.size() + (new_count - old_count));
    pos.resize(new_count, kNotFree);
    for (std::size_t c = old_count; c < new_count; ++c)
    {
        pos[c] = ids.size();
        ids.push_back(static_cast<ChunkId>(c));
    }
}

void FreeList::seed(std::size_t count)
{
    // Ascending pop order: ids ends [count-1 .. 0] so the back is 0
    // and the first pop hands out the smallest id — deterministic,
    // not required.
    if (count > static_cast<std::size_t>(kNotFree))
    {
        throw std::runtime_error(
            "free list: seeding " + std::to_string(count) +
            " ids overflows the ChunkId space (kNotFree=" +
            std::to_string(kNotFree) + ")");
    }
    ids.clear();
    ids.reserve(count);
    pos.assign(count, kNotFree);
    for (std::size_t i = count; i-- > 0;)
    {
        pos[i] = static_cast<ChunkId>(ids.size());
        ids.push_back(static_cast<ChunkId>(i));
    }
}

std::optional<ChunkId> FreeList::pop()
{
    if (ids.empty())
    {
        return std::nullopt;
    }
    ChunkId const c = ids.back();
    ids.pop_back();
    pos[c] = kNotFree;
    return c;
}

void FreeList::push(ChunkId c)
{
    if (static_cast<std::size_t>(c) >= pos.size())
    {
        throw std::runtime_error("free list: push(" + std::to_string(c) +
            ") names an id outside the seeded space (pos holds " +
            std::to_string(pos.size()) + ")");
    }
    if (pos[c] != kNotFree)
    {
        throw std::runtime_error("free list: push(" + std::to_string(c) +
            ") names an id already free (duplicate member)");
    }
    pos[c] = static_cast<ChunkId>(ids.size());
    ids.push_back(c);
}

void FreeList::remove(ChunkId c)
{
    if (static_cast<std::size_t>(c) >= pos.size())
    {
        throw std::runtime_error("free list: remove(" + std::to_string(c) +
            ") names an id outside the seeded space (pos holds " +
            std::to_string(pos.size()) + ")");
    }
    if (pos[c] == kNotFree)
    {
        throw std::runtime_error("free list: remove(" + std::to_string(c) +
            ") names an id not free (non-member)");
    }
    // Swap-with-back: the last id moves into c's slot, both position
    // entries update, c leaves — O(1) whatever c's place in ids.
    std::size_t const at = pos[c];
    ChunkId const last = ids.back();
    ids[at] = last;
    pos[last] = static_cast<ChunkId>(at);
    ids.pop_back();
    pos[c] = kNotFree;
}

bool FreeList::member(ChunkId c) const
{
    if (static_cast<std::size_t>(c) >= pos.size())
    {
        throw std::runtime_error("free list: member(" + std::to_string(c) +
            ") names an id outside the seeded space (pos holds " +
            std::to_string(pos.size()) + ")");
    }
    return pos[c] != kNotFree;
}

std::size_t FreeList::size() const
{
    return ids.size();
}

// ---- the reverse map's key helpers ----

bool operator==(LogicalChunk const &a, LogicalChunk const &b)
{
    return a.tile == b.tile && a.index == b.index;
}

std::size_t LogicalChunkHash::operator()(LogicalChunk const &lc) const noexcept
{
    // The pair packed to 64 bits, then a splitmix64 finalizer: plain
    // sequential (tile, index) patterns spread across buckets.
    std::uint64_t k = (static_cast<std::uint64_t>(lc.tile) << 32) |
        static_cast<std::uint64_t>(lc.index);
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return static_cast<std::size_t>(k);
}

// ---- the range guards (the contract's loud misuse surface) ----

BindingTable::Segment &BindingTable::segment_of(NodeId n, char const *act)
{
    if (static_cast<std::size_t>(n) >= seg_.size())
    {
        throw std::runtime_error(
            "binding table: " + std::string(act) + " names node " +
            std::to_string(n) + ", the table holds " +
            std::to_string(seg_.size()) + " segments");
    }
    return seg_[static_cast<std::size_t>(n)];
}

BindingTable::Segment const &BindingTable::segment_of(
    NodeId n, char const *act) const
{
    if (static_cast<std::size_t>(n) >= seg_.size())
    {
        throw std::runtime_error(
            "binding table: " + std::string(act) + " names node " +
            std::to_string(n) + ", the table holds " +
            std::to_string(seg_.size()) + " segments");
    }
    return seg_[static_cast<std::size_t>(n)];
}

BindingTable::Segment &BindingTable::chunk_of(
    NodeId n, ChunkId c, char const *act)
{
    Segment &s = segment_of(n, act);
    if (static_cast<std::size_t>(c) >= s.owner_tile_.size())
    {
        throw std::runtime_error(
            "binding table: " + std::string(act) + " names chunk " +
            std::to_string(c) + " of node " + std::to_string(n) +
            ", whose pool holds " + std::to_string(s.owner_tile_.size()) +
            " chunks");
    }
    return s;
}

BindingTable::Segment const &BindingTable::chunk_of(
    NodeId n, ChunkId c, char const *act) const
{
    Segment const &s = segment_of(n, act);
    if (static_cast<std::size_t>(c) >= s.owner_tile_.size())
    {
        throw std::runtime_error(
            "binding table: " + std::string(act) + " names chunk " +
            std::to_string(c) + " of node " + std::to_string(n) +
            ", whose pool holds " + std::to_string(s.owner_tile_.size()) +
            " chunks");
    }
    return s;
}

// ---- init: N segments, all free  ----

BindingTable::BindingTable(
    std::vector<std::size_t> const &pool_cardinalities)
{
    init(pool_cardinalities);
}

void BindingTable::init(std::vector<std::size_t> const &pool_cardinalities)
{
    // The addressability guard first, before any segment allocates: a
    // cardinality beyond kNotFree cannot be seeded (the sentinel would
    // collide with a real id), and the dense arrays of such a pool
    // must never even be attempted.
    for (std::size_t cardinality : pool_cardinalities)
    {
        if (cardinality > static_cast<std::size_t>(FreeList::kNotFree))
        {
            throw std::runtime_error(
                "binding table: init pool cardinality " +
                std::to_string(cardinality) +
                " overflows the ChunkId space (kNotFree=" +
                std::to_string(FreeList::kNotFree) + ")");
        }
    }
    seg_.assign(pool_cardinalities.size(), Segment{});
    for (std::size_t n = 0; n < pool_cardinalities.size(); ++n)
    {
        std::size_t const cardinality = pool_cardinalities[n];
        Segment &s = seg_[n];
        s.owner_tile_.assign(cardinality, kNoTile);
        s.owner_index_.assign(cardinality, 0);
        s.flag_.assign(cardinality, 0);
        s.rev_.clear();
        s.free_.seed(cardinality); // throws loudly on an unaddressable
                                   // cardinality
    }
}

// ---- the fold's transitions  ----

void BindingTable::grow_node(NodeId n, std::size_t new_cardinality)
{
    // The growth act: extend node n's segment to
    // new_cardinality slots, every appended slot FREE (no owner, no
    // flag, on the free list in ascending id order — the same state a
    // fresh init seeds). Existing slots, bindings, and flags are
    // untouched: growth only ever ADDS free chunks (extends the
    // boundary's free-slot set). Shrinking is refused loudly (a pool
    // never gives chunks back); a cardinality beyond kNotFree is the
    // same overflow init refuses.
    Segment &s = segment_of(n, "grow_node");
    if (new_cardinality < s.owner_tile_.size())
    {
        throw std::runtime_error(
            "binding table: grow_node(" + std::to_string(n) + ", " +
            std::to_string(new_cardinality) +
            ") would shrink the pool of " +
            std::to_string(s.owner_tile_.size()) + " chunks — refused");
    }
    if (new_cardinality > static_cast<std::size_t>(FreeList::kNotFree))
    {
        throw std::runtime_error(
            "binding table: grow_node(" + std::to_string(n) + ", " +
            std::to_string(new_cardinality) +
            ") overflows the ChunkId space (kNotFree=" +
            std::to_string(FreeList::kNotFree) + ")");
    }
    std::size_t const old_card = s.owner_tile_.size();
    if (new_cardinality == old_card)
    {
        return;
    }
    s.owner_tile_.resize(new_cardinality, kNoTile);
    s.owner_index_.resize(new_cardinality, 0);
    s.flag_.resize(new_cardinality, 0);
    s.free_.grow(old_card, new_cardinality);
}

void BindingTable::link(NodeId n, ChunkId c, LogicalChunk lc)
{
    Segment &s = chunk_of(n, c, "link");

    // The structural guards first — a throw must leave the table
    // untouched for the census. Both fire only for a
    // caller that skipped the checker's rows (the header contract):
    // the reverse map must keep naming exactly the linked bindings,
    // and the free list's remove throws on a non-member.
    auto held = s.rev_.find(lc);
    if (held != s.rev_.end())
    {
        ChunkId const old = held->second;
        throw std::runtime_error(
            "binding table: link names logical chunk (" +
            std::to_string(lc.tile) + "," + std::to_string(lc.index) +
            ") already bound on node " + std::to_string(n) + " (chunk " +
            std::to_string(old) + ") -- the reverse map stays exact "
            "; the fold checks binding_of first");
    }
    s.free_.remove(c); // loud iff c is already linked (double-link
                       // misuse; a passing fold linked nothing there)

    // The pure O(1) move: the binding starts OUTDATED —
    // it holds nothing until a producer actualizes it.
    s.owner_tile_[static_cast<std::size_t>(c)] = lc.tile;
    s.owner_index_[static_cast<std::size_t>(c)] = lc.index;
    s.flag_[static_cast<std::size_t>(c)] = 0;
    s.rev_.emplace(lc, c);
}

void BindingTable::unlink(NodeId n, ChunkId c)
{
    Segment &s = chunk_of(n, c, "unlink");

    // The free-list discipline first: push throws on a duplicate
    // member — an already-free chunk (unlink-free misuse) — before any
    // mutation, so a throw leaves the table untouched.
    s.free_.push(c);

    // The pure O(1) move: T[n, c] := free; the flag
    // drops with the binding. The reverse entry dies only if it names
    // c — it always does in contract (the fold checked the binding's
    // identity), and the guard keeps the map exact for out-of-contract
    // callers. UNLINK touches no replica.
    TileId const tile = s.owner_tile_[static_cast<std::size_t>(c)];
    if (tile != kNoTile)
    {
        LogicalChunk const lc{tile, s.owner_index_[static_cast<std::size_t>(c)]};
        auto it = s.rev_.find(lc);
        if (it != s.rev_.end() && it->second == c)
        {
            s.rev_.erase(it);
        }
    }
    s.owner_tile_[static_cast<std::size_t>(c)] = kNoTile;
    s.owner_index_[static_cast<std::size_t>(c)] = 0;
    s.flag_[static_cast<std::size_t>(c)] = 0;
}

void BindingTable::actualize(NodeId n, ChunkId c)
{
    Segment &s = chunk_of(n, c, "actualize");
    // The pure O(1) move: := actual. The fold actualizes
    // linked bindings only (the rows ran first); on a free slot the
    // write is meaningless and wiped at the next link's reset.
    s.flag_[static_cast<std::size_t>(c)] = 1;
}

void BindingTable::outdate_replicas(LogicalChunk lc, NodeId kn, ChunkId kc)
{
    // The writer's own (node, chunk) is range-checked like every act
    // (programmer misuse is loud) and then skipped: the writer's own
    // binding is never its own victim.
    (void) chunk_of(kn, kc, "outdate_replicas");
    for (std::size_t n = 0; n < seg_.size(); ++n)
    {
        auto it = seg_[n].rev_.find(lc);
        if (it == seg_[n].rev_.end())
        {
            continue; // no replica of lc on this node
        }
        if (n == static_cast<std::size_t>(kn) && it->second == kc)
        {
            continue; // the writer itself
        }
        // := outdated, actual replicas included: a
        // write supersedes every other replica's bytes.
        seg_[n].flag_[static_cast<std::size_t>(it->second)] = 0;
    }
}

// ---- the queries (the state read, all O(1)) ----

bool BindingTable::linked(NodeId n, ChunkId c) const
{
    Segment const &s = chunk_of(n, c, "linked");
    return s.owner_tile_[static_cast<std::size_t>(c)] != kNoTile;
}

std::optional<std::pair<LogicalChunk, bool>> BindingTable::at(
    NodeId n, ChunkId c) const
{
    Segment const &s = chunk_of(n, c, "at");
    std::size_t const at_c = static_cast<std::size_t>(c);
    if (s.owner_tile_[at_c] == kNoTile)
    {
        return std::nullopt;
    }
    return std::make_pair(
        LogicalChunk{s.owner_tile_[at_c], s.owner_index_[at_c]},
        s.flag_[at_c] != 0);
}

bool BindingTable::flag_actual(NodeId n, ChunkId c) const
{
    Segment const &s = chunk_of(n, c, "flag_actual");
    if (s.owner_tile_[static_cast<std::size_t>(c)] == kNoTile)
    {
        // The flag is a BINDING's property; a free chunk
        // has none. The fold reads flags only after its linkage checks.
        throw std::runtime_error(
            "binding table: flag_actual reads node " + std::to_string(n) +
            " chunk " + std::to_string(c) + ", which is free -- the flag "
            "is a binding's property");
    }
    return s.flag_[static_cast<std::size_t>(c)] != 0;
}

std::optional<ChunkId> BindingTable::binding_of(
    NodeId n, LogicalChunk lc) const
{
    Segment const &s = segment_of(n, "binding_of");
    auto it = s.rev_.find(lc);
    if (it == s.rev_.end())
    {
        return std::nullopt;
    }
    return it->second;
}

bool BindingTable::free(NodeId n, ChunkId c) const
{
    // The free VIEW's own membership (free_.member), not !linked
    // re-derived: the state read asks the free lists,
    // and the view and the dense columns must agree.
    Segment const &s = chunk_of(n, c, "free");
    return s.free_.member(c);
}

std::optional<ChunkId> BindingTable::pop_free(NodeId n) const
{
    Segment const &s = segment_of(n, "pop_free");
    if (s.free_.ids.empty())
    {
        // Nullopt, not a fault: exhaustion is the planner's census.
        return std::nullopt;
    }
    return s.free_.ids.back(); // the next handout, a query
}

std::optional<ChunkId> BindingTable::pop_free_run(
    NodeId n, std::size_t k) const
{
    // The contiguous-run handout (a QUERY like pop_free — the removal
    // happens at link, one chunk at a time): the LOWEST run of k
    // consecutive free chunk ids of the node's pool, nullopt when the
    // free list carries no full run (the caller falls back to
    // per-chunk pops). One pass over the cardinality through the free
    // view's position index — O(C), never a sort; deterministic (the
    // lowest run, the scan's own order).
    Segment const &s = segment_of(n, "pop_free_run");
    if (k == 0)
    {
        return std::nullopt;
    }
    std::size_t const card = s.owner_tile_.size();
    if (k == 1)
    {
        // A run of one is the LOWEST free chunk, never the pop
        // order's back: the lowest free chunk is by construction a
        // RUN EDGE (its left neighbor is bound), so single-chunk
        // tiles consume edges and the interiors of long free runs
        // survive for the multi-chunk tiles whose codelets pay the
        // staged path when scattered. The blind LIFO pop shreds runs
        // from the middle, fragmenting the pool for multi-chunk tiles.
        for (std::size_t c = 0; c < card; ++c)
        {
            if (s.free_.pos[c] != FreeList::kNotFree)
            {
                return static_cast<ChunkId>(c);
            }
        }
        return std::nullopt;
    }
    std::size_t run = 0;
    for (std::size_t c = 0; c < card; ++c)
    {
        if (s.free_.pos[c] != FreeList::kNotFree)
        {
            run += 1;
            if (run == k)
            {
                return static_cast<ChunkId>(c + 1u - k);
            }
        }
        else
        {
            run = 0;
        }
    }
    return std::nullopt;
}

std::size_t BindingTable::free_count(NodeId n) const
{
    return segment_of(n, "free_count").free_.size();
}

std::size_t BindingTable::longest_free_run(NodeId n) const
{
    // The fragmentation meter's one query (diagnosis only): the length
    // of the longest run of consecutive free chunk ids — the number a
    // multi-chunk contiguous handout could have taken at this moment.
    Segment const &s = segment_of(n, "longest_free_run");
    std::size_t const card = s.owner_tile_.size();
    std::size_t best = 0;
    std::size_t run = 0;
    for (std::size_t c = 0; c < card; ++c)
    {
        if (s.free_.pos[c] != FreeList::kNotFree)
        {
            run += 1;
            if (run > best)
            {
                best = run;
            }
        }
        else
        {
            run = 0;
        }
    }
    return best;
}

std::size_t BindingTable::node_count() const
{
    return seg_.size();
}

std::size_t BindingTable::pool_cardinality(NodeId n) const
{
    return segment_of(n, "pool_cardinality").owner_tile_.size();
}

// ---- the boundary halves  ----

bool BindingTable::equals(std::vector<EntryBinding> const &entry) const
{
    // Sparse-versus-dense over ALL nodes' segments, binding for
    // binding, flag for flag, LINEAR in the one measure — O(C +
    // |entry|), never a sort. The walk demands the entry
    // STRICTLY ASCENDING by (node, chunk) — the plan's own entry
    // convention (Plan.entry is sorted by (node, chunk)) — so an
    // unsorted or duplicate-carrying vector is itself a difference,
    // false like any other. Any difference, an unknown node or chunk
    // included, is false: boundary-mismatch / boundary-discontinuity
    // are the caller's censuses (never a throw here).
    std::size_t linked_total = 0;
    for (std::size_t n = 0; n < seg_.size(); ++n)
    {
        for (TileId owner : seg_[n].owner_tile_)
        {
            if (owner != kNoTile)
            {
                linked_total += 1;
            }
        }
    }
    if (entry.size() != linked_total)
    {
        return false; // a linked slot unlisted, or a phantom listed
    }

    // One walk, two checks per entry: the slot match (in range,
    // linked, tile, index, flag) and the strict ascent. With the
    // totals equal, the DISTINCT ascending (node, chunk) keys that
    // each match a linked slot cover every linked slot — a bijection
    // by counting, no scratch and no per-node tally needed (a binding
    // moved between nodes breaks the slot match, the ascent, or the
    // total).
    for (std::size_t i = 0; i < entry.size(); ++i)
    {
        EntryBinding const &b = entry[i];
        if (static_cast<std::size_t>(b.node) >= seg_.size())
        {
            return false; // an unknown node is a difference
        }
        Segment const &s = seg_[static_cast<std::size_t>(b.node)];
        std::size_t const at_c = static_cast<std::size_t>(b.chunk);
        if (at_c >= s.owner_tile_.size() || s.owner_tile_[at_c] == kNoTile ||
            s.owner_tile_[at_c] != b.tile ||
            s.owner_index_[at_c] != b.index ||
            (s.flag_[at_c] != 0) != b.actual)
        {
            return false;
        }
        if (i > 0)
        {
            EntryBinding const &prev = entry[i - 1];
            if (prev.node > b.node ||
                (prev.node == b.node && prev.chunk >= b.chunk))
            {
                return false; // unsorted or a duplicate slot named
                              // twice — a difference either way
            }
        }
    }
    return true;
}

std::vector<EntryBinding> BindingTable::snapshot() const
{
    // The exit fold: every linked binding with its flag,
    // sorted by (node, chunk) — the walk's own order, no sort needed.
    std::vector<EntryBinding> out;
    for (std::size_t n = 0; n < seg_.size(); ++n)
    {
        for (std::size_t c = 0; c < seg_[n].owner_tile_.size(); ++c)
        {
            if (seg_[n].owner_tile_[c] == kNoTile)
            {
                continue;
            }
            EntryBinding b;
            b.node = static_cast<NodeId>(n);
            b.chunk = static_cast<ChunkId>(c);
            b.tile = seg_[n].owner_tile_[c];
            b.index = seg_[n].owner_index_[c];
            b.actual = seg_[n].flag_[c] != 0;
            out.push_back(b);
        }
    }
    return out;
}

std::vector<SegmentSnapshot> BindingTable::render() const
{
    // The census's per-node form: each
    // segment's bindings ascending by ChunkId with the free list
    // printed beside it — data for the emitter, which aborts.
    std::vector<SegmentSnapshot> out;
    out.reserve(seg_.size());
    for (std::size_t n = 0; n < seg_.size(); ++n)
    {
        SegmentSnapshot ss;
        ss.node = static_cast<NodeId>(n);
        ss.pool = seg_[n].owner_tile_.size();
        for (std::size_t c = 0; c < seg_[n].owner_tile_.size(); ++c)
        {
            if (seg_[n].owner_tile_[c] == kNoTile)
            {
                continue;
            }
            EntryBinding b;
            b.node = ss.node;
            b.chunk = static_cast<ChunkId>(c);
            b.tile = seg_[n].owner_tile_[c];
            b.index = seg_[n].owner_index_[c];
            b.actual = seg_[n].flag_[c] != 0;
            ss.bindings.push_back(b);
        }
        ss.free_list = seg_[n].free_.ids; // the mechanism's own order
        out.push_back(std::move(ss));
    }
    return out;
}

} // namespace nnhaul
