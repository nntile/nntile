/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * one binding table of the chunk-graph contract: per node, ChunkId to
 * (logical chunk, flag) or free, with the free lists as derived views.
 *
 * @file src/binding_table.hh
 * The N-segment binding table: one table, one flag column, N segments —
 * one per node — as a per-node structure of arrays with O(1)
 * transitions, the free list as a derived view, the reverse map from
 * logical chunk to bound chunk, and the boundary halves (equals over
 * ALL nodes' segments, snapshot as the exit fold). One table collapses
 * what the previous engine kept as a chunk table, an ownership record,
 * a host table, and two free lists.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

#include "plan.hh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nnhaul
{

// ---- the free list: the derived view's mechanism  ----

// The per-node free set as an intrusive vector plus a position index:
// ids holds the free ids with the back the next pop, pos holds per
// ChunkId its slot in ids or kNotFree. pop, push, and remove are O(1);
// seed plants ascending pop order (the blocks are interchangeable, only
// deterministically ordered). Loud on misuse: push on a duplicate
// member and remove on a non-member throw (std::runtime_error, never
// serviced); an id outside the seeded space is programmer misuse and
// throws likewise.
struct FreeList // O(1) pop, push, remove, membership
{
    std::vector<ChunkId> ids; // the free ids; the back is the next pop
    std::vector<ChunkId> pos; // per ChunkId: its slot in ids, or kNotFree
    static constexpr ChunkId kNotFree = std::numeric_limits<ChunkId>::max();

    void seed(std::size_t count);       // ascending pop order
    void grow(std::size_t old_count,
        std::size_t new_count);        // append [old,new) free, ascending
    std::optional<ChunkId> pop();       // nullopt when empty
    void push(ChunkId c);               // throws on a duplicate member
    void remove(ChunkId c);             // swap-with-back; throws on a
                                        // non-member
    bool member(ChunkId c) const;       // O(1) membership; throws on an
                                        // id outside the seeded space
    std::size_t size() const;           // the free count, O(1)
};

// ---- the free marker and the reverse map's key helpers ----

// owner_tile_'s free marker: a slot whose owner tile
// is kNoTile is free. TileId is a dense small integer, so the table
// leaves the extreme value to the marker; the tile table is sparse and
// a plan naming kNoTile as a tile simply never checks against it.
inline constexpr TileId kNoTile = std::numeric_limits<TileId>::max();

// The reverse map's key equality. plan.hh owns the type; the equality
// belongs to the key's use here.
bool operator==(LogicalChunk const &a, LogicalChunk const &b);

// The reverse map's hash functor: the pair
// packed to 64 bits and run through a splitmix64 finalizer — dense
// bucket distribution for sequential tile/index patterns.
struct LogicalChunkHash
{
    std::size_t operator()(LogicalChunk const &lc) const noexcept;
};

// ---- the binding table  ----

// One table, one flag column, N segments: per node,
// ChunkId -> (logical chunk, flag) or free. The free list is a DERIVED
// VIEW of the table — never a second table to keep in sync —
// maintained by the same O(1) transitions.
//
// CONTRACT, stated once (the same calls are made by the checker, the
// state read, and the executor — one pure transition layer, three
// callers):
//
// - The fold's transitions perform NO validation. The per-op checks
//   (double-link, double-binding, unlink-free, identity-mismatch,
//   copy-*, read-outdated, ...) are the CHECKER's, evaluated before the
//   effect call; the planner's state read folds ops it built; the
//   executor applies checked ops and re-evaluates the checks
//   abort-only. In contract, every transition below is a pure O(1)
//   move.
//
// - Programmer range misuse throws loudly (std::runtime_error, never
//   serviced): a node not of the table's N segments, or a chunk not of
//   its node's pool — the caller's defect, not a fold fault.
//
// - The FreeList's own discipline stays loud: link removes from the
//   free list and throws on a non-member (an already-linked chunk),
//   unlink pushes and throws on a duplicate member (an already-free
//   chunk). A caller that skipped the checker's checks therefore fails
//   structurally loud instead of corrupting silently; a passing fold
//   never reaches these throws. One further structural guard of the
//   same kind: link naming a logical chunk already bound on that node
//   throws — the reverse map must keep naming exactly the linked
//   bindings, and the overwrite would orphan the old chunk's entry.
//
// - pop_free on an exhausted pool is nullopt, NOT a fault: exhaustion
//   is the planner's pool-exhaustion census at the state read, never
//   this table's throw.
//
// - equals() never throws: ANY difference across ALL nodes' segments —
//   an out-of-range, unsorted, or duplicate entry binding included —
//   is false; boundary-mismatch / boundary-discontinuity are the
//   caller's censuses.
class BindingTable
{
  public:
    BindingTable() = default;

    // The init-time act: N segments — one per pool
    // cardinality, indexed by NodeId — ALL FREE; the initial load then
    // fills the table through the same transitions.
    // Re-callable: a second init rebuilds the all-free table. Throws
    // loudly on a cardinality the ChunkId space cannot address.
    explicit BindingTable(std::vector<std::size_t> const &pool_cardinalities);

    void init(std::vector<std::size_t> const &pool_cardinalities);
    // all free ; the constructor is this act

    // ---- the fold's transitions  ----

    void link(NodeId n, ChunkId c, LogicalChunk lc); // := (lc, outdated)
                                                     //
    void unlink(NodeId n, ChunkId c);                // := free; the flag
                                                     // drops with the
                                                     // binding
    void actualize(NodeId n, ChunkId c);             // := actual
                                                     //
    void outdate_replicas(LogicalChunk lc, NodeId kn, ChunkId kc);
    // Every OTHER linked replica of lc across all N segments :=
    // outdated, actual ones included; the writer (kn,
    // kc) itself is never touched. Loops the N reverse lookups —
    // bounded by the node count, a run-scoped constant.

    // ---- queries, all O(1) ----

    bool linked(NodeId n, ChunkId c) const; // false iff free

    std::optional<std::pair<LogicalChunk, bool>> at(
        NodeId n, ChunkId c) const; // the binding and its flag, or
                                    // nullopt when free

    bool flag_actual(NodeId n, ChunkId c) const; // the binding's flag;
                                                 // throws on a free
                                                 // chunk — the flag is
                                                 // a binding's property

    std::optional<ChunkId> binding_of(
        NodeId n, LogicalChunk lc) const; // the reverse lookup: the
                                          // node's chunk bound to lc,
                                          // if any

    bool free(NodeId n, ChunkId c) const; // the free VIEW's membership
                                          // (the free list itself, not
                                          // !linked re-derived — the
                                          // two must agree)

    // Extend node n's pool (see the .cc's note):
    // appended slots are free, existing state untouched.
    void grow_node(NodeId n, std::size_t new_cardinality);

    std::optional<ChunkId> pop_free(NodeId n) const;
    // The free list's next handout (the back), a QUERY: the removal
    // happens at link's transition — the free list is a derived view,
    // never popped apart from the table. Nullopt when the
    // node's pool is exhausted.

    // The contiguous-run handout, a QUERY of the same kind: the LOWEST
    // run of k consecutive free chunk ids of node n's pool, nullopt
    // when no full run sits free (the caller falls back to per-chunk
    // pops). The consumers' contiguity fast paths (the codelets'
    // panel/descriptor chaining) ride on consecutive chunk ids; the
    // pop discipline — WHICH free chunk a link names — is the
    // planner's own freedom (the blocks are interchangeable,
    // deterministically ordered), and this discipline spends
    // it on runs. O(pool cardinality), one pass, never a sort.
    std::optional<ChunkId> pop_free_run(NodeId n, std::size_t k) const;

    std::size_t free_count(NodeId n) const; // the node's free count

    // The longest run of consecutive free chunk ids (diagnosis only:
    // the fragmentation meter behind the planner's run-miss stat).
    std::size_t longest_free_run(NodeId n) const;

    std::size_t node_count() const;         // N, the segment count

    std::size_t pool_cardinality(NodeId n) const; // the fixed pool

    // ---- the boundary halves  ----

    // The table against an entry state, across ALL nodes' segments,
    // binding for binding, flag for flag: true iff the entry names
    // exactly the linked slots, STRICTLY ASCENDING by (node, chunk) —
    // the plan's own entry convention — and every field
    // agrees; an unsorted or duplicate-carrying vector is itself a
    // difference. Linear in the one measure, O(C + |entry|), never a
    // sort. Never throws — any difference (an unknown
    // node or chunk included) is false, the caller's census.
    bool equals(std::vector<EntryBinding> const &entry) const;

    // The exit fold : the linked bindings with their
    // flags, sorted by (node, chunk) — the successor plan's entry
    // state and the chaining check's input.
    std::vector<EntryBinding> snapshot() const;

    // The census render : one
    // SegmentSnapshot per node — the segment's bindings ascending by
    // ChunkId with the free list printed beside it.
    std::vector<SegmentSnapshot> render() const;

  private:
    struct Segment
    {
        std::vector<TileId> owner_tile_;   // kNoTile marks free; indexed
                                           // by ChunkId
        std::vector<ChunkIndex> owner_index_;
        std::vector<std::uint8_t> flag_;   // 0 outdated, 1 actual
        std::unordered_map<LogicalChunk, ChunkId, LogicalChunkHash>
            rev_;                          // logical chunk -> bound chunk
        FreeList free_;
    };
    std::vector<Segment> seg_; // indexed by NodeId; N segments, one
                               // table

    // Range guards (the loud programmer-misuse surface of the
    // contract): node not of the N segments, chunk not of its pool.
    Segment &segment_of(NodeId n, char const *act);
    Segment const &segment_of(NodeId n, char const *act) const;
    Segment &chunk_of(NodeId n, ChunkId c, char const *act);
    Segment const &chunk_of(NodeId n, ChunkId c, char const *act) const;
};

} // namespace nnhaul
