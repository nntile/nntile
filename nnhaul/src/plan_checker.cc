/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine. This file is the
 * stateless plan checker's one fold: the entry-state check, the
 * chaining comparison, the four per-op dispatch tables, the
 * correspondence scan, and the census render.
 *
 * @file src/plan_checker.cc
 * The fold behind src/plan_checker.hh: the BindingTable built
 * from the inventory's cardinalities; the entry-state check before op
 * 0 (unknown-id, unknown-tile, index-out-of-range, double-binding,
 * double-link) folding each validated binding in — T.link, then
 * T.actualize for an actual entry; the chaining check T.snapshot() ==
 * *chain over ALL nodes, else boundary-discontinuity;
 * the fold over the queue through fold_link, fold_unlink, fold_copy,
 * fold_tileop — every row in its fixed group order (shape,
 * references, identity/linkage, topology, flags, within-op
 * distinctness), every row before any effect, every fault name
 * verbatim; the correspondence scan over the same walk
 * (correspondence_step per TILE_OP, correspondence_close at the end)
 * with its rules applied in order; and the census:
 * the fold rendered per node with the free list beside it, the entry
 * state, the ops prefix, the inputs' relevant facts. One pass,
 * O(ops + operands + C + |entry|) — no sort, no rescan.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "plan_checker.hh"

#include "binding_table.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nnhaul
{

namespace
{

// ---- the caller-defect surface: loud, never a plan fault ----

void require(void const *p, char const *what)
{
    if (p == nullptr)
    {
        throw std::runtime_error(
            std::string("plan_checker: the check input carries no ") +
            what + " ");
    }
}

// Dense packings for the per-op distinctness rows and the
// correspondence accumulators — O(1) hash keys, no sorts anywhere.
std::uint64_t chunk_key(NodeId n, ChunkId c)
{
    return (static_cast<std::uint64_t>(n) << 32) |
        static_cast<std::uint64_t>(c);
}

std::uint64_t logical_key(TileId t, ChunkIndex i)
{
    return (static_cast<std::uint64_t>(t) << 32) |
        static_cast<std::uint64_t>(i);
}

// ---- the finding: one violated row, positioned ----

struct Finding
{
    std::string fault;         // the fault's name, verbatim
    CheckSite site = CheckSite::Entry;
    std::size_t position = 0;  // queue index, or entry binding index
    std::optional<PlanOp> op;  // the offending op verbatim
    std::string detail;        // census-only precision, never the name
};

Finding at_queue(char const *fault, std::size_t p, PlanOp const &op)
{
    Finding f;
    f.fault = fault;
    f.site = CheckSite::Queue;
    f.position = p;
    f.op = op;
    return f;
}

Finding at_entry(
    char const *fault, std::size_t binding, std::string detail)
{
    Finding f;
    f.fault = fault;
    f.site = CheckSite::Entry;
    f.position = binding;
    f.detail = std::move(detail);
    return f;
}

Finding at_close(
    char const *fault, std::string detail, PlanOp const *witness = nullptr,
    std::size_t p = 0)
{
    Finding f;
    f.fault = fault;
    f.site = CheckSite::Close;
    f.detail = std::move(detail);
    if (witness != nullptr)
    {
        f.op = *witness;
        f.position = p; // the census witness's place, informative
    }
    return f;
}

// ---- the registry views the fold reads ----

struct Ctx
{
    NodeInventory const *inv;
    TileTable const *tiles;
    ExecutabilityTable const *runnable;
    // TILE_OP operands whose binding holds another logical chunk. The
    // close correspondence faults take precedence, so these are
    // reported after it (a rejected plan either way, but the most
    // specific fault is reported first).
    std::vector<std::pair<std::size_t, PlanOp>> *identity = nullptr;
};

// ---- fold_link  ----

std::optional<Finding> fold_link(
    PlanOp const &op, std::size_t p, BindingTable &t, Ctx const &ctx)
{
    // Row order fixed; all rows before the effect.
    if (!op.operands.empty()) // row 1: shape — exactly the four fields
    {
        return at_queue("malformed-plan", p, op);
    }
    if (!ctx.inv->chunk_valid(op.u.link.node,
            op.u.link.chunk)) // row 2: ids
    {
        return at_queue("unknown-id", p, op);
    }
    if (!ctx.tiles->carries(op.u.link.tile)) // row 3: tile
    {
        return at_queue("unknown-tile", p, op);
    }
    if (op.u.link.index >=
        ctx.tiles->n(op.u.link.tile)) // row 4: index
    {
        return at_queue("index-out-of-range", p, op);
    }
    if (t.linked(op.u.link.node, op.u.link.chunk)) // row 5: free
    {
        return at_queue("double-link", p, op);
    }
    LogicalChunk const lc{op.u.link.tile, op.u.link.index};
    if (t.binding_of(op.u.link.node, lc)
            .has_value()) // row 6: one binding per node
    {
        return at_queue("double-binding", p, op);
    }
    // Effect: := (lc, outdated) — the binding starts OUTDATED; a LINK
    // moves no bytes and actualizes nothing.
    t.link(op.u.link.node, op.u.link.chunk, lc);
    return std::nullopt;
}

// ---- fold_unlink  ----

std::optional<Finding> fold_unlink(
    PlanOp const &op, std::size_t p, BindingTable &t, Ctx const &ctx)
{
    // Rows 1-4 as LINK: shape, ids, tile, index.
    if (!op.operands.empty())
    {
        return at_queue("malformed-plan", p, op);
    }
    if (!ctx.inv->chunk_valid(op.u.unlink.node, op.u.unlink.chunk))
    {
        return at_queue("unknown-id", p, op);
    }
    if (!ctx.tiles->carries(op.u.unlink.tile))
    {
        return at_queue("unknown-tile", p, op);
    }
    if (op.u.unlink.index >= ctx.tiles->n(op.u.unlink.tile))
    {
        return at_queue("index-out-of-range", p, op);
    }
    if (!t.linked(op.u.unlink.node, op.u.unlink.chunk)) // row 5: bound
    {
        return at_queue("unlink-free", p, op);
    }
    LogicalChunk const lc{op.u.unlink.tile, op.u.unlink.index};
    auto held = t.at(op.u.unlink.node, op.u.unlink.chunk);
    if (!held.has_value() ||
        !(held->first == lc)) // row 6: the binding's own identity
    {
        return at_queue("identity-mismatch", p, op);
    }
    // Effect: := free; the flag drops with the binding. UNLINK is
    // indifferent to the flag and touches no replica.
    t.unlink(op.u.unlink.node, op.u.unlink.chunk);
    return std::nullopt;
}

// ---- fold_copy  ----

std::optional<Finding> fold_copy(
    PlanOp const &op, std::size_t p, BindingTable &t, Ctx const &ctx)
{
    if (!op.operands.empty()) // row 1: shape — exactly the four fields
    {
        return at_queue("malformed-plan", p, op);
    }
    // Row 2: src's and dst's (node, chunk) of the pools, src first.
    if (!ctx.inv->chunk_valid(op.u.copy.src.node, op.u.copy.src.chunk) ||
        !ctx.inv->chunk_valid(op.u.copy.dst.node, op.u.copy.dst.chunk))
    {
        return at_queue("unknown-id", p, op);
    }
    // Rows 3-4 as LINK: tile, index.
    if (!ctx.tiles->carries(op.u.copy.tile))
    {
        return at_queue("unknown-tile", p, op);
    }
    if (op.u.copy.index >= ctx.tiles->n(op.u.copy.tile))
    {
        return at_queue("index-out-of-range", p, op);
    }
    LogicalChunk const lc{op.u.copy.tile, op.u.copy.index};
    // Rows 5-6: each endpoint linked exactly to (tile, index) — a free
    // endpoint is a mismatch like any other.
    auto src = t.at(op.u.copy.src.node, op.u.copy.src.chunk);
    if (!src.has_value() || !(src->first == lc))
    {
        return at_queue("copy-identity-mismatch", p, op);
    }
    auto dst = t.at(op.u.copy.dst.node, op.u.copy.dst.chunk);
    if (!dst.has_value() || !(dst->first == lc))
    {
        return at_queue("copy-identity-mismatch", p, op);
    }
    // Row 7: a channel needs two distinct endpoints.
    if (op.u.copy.src.node == op.u.copy.dst.node)
    {
        return at_queue("copy-same-node", p, op);
    }
    // Row 8: the staging rule's check-time half — the inventory is
    // static, so an absent channel is complete.
    if (!ctx.inv->has_channel(op.u.copy.src.node, op.u.copy.dst.node))
    {
        return at_queue("copy-no-channel", p, op);
    }
    // Rows 9-10: the flag halves — copies never consume and never
    // produce a new version.
    if (!t.flag_actual(op.u.copy.src.node, op.u.copy.src.chunk))
    {
        return at_queue("copy-from-outdated", p, op);
    }
    if (t.flag_actual(op.u.copy.dst.node, op.u.copy.dst.chunk))
    {
        return at_queue("copy-to-actual", p, op);
    }
    // Effect: bytes move at execution; the fold verifies flags, never
    // bytes  — T[dst].flag := actual, no other change.
    t.actualize(op.u.copy.dst.node, op.u.copy.dst.chunk);
    return std::nullopt;
}

// ---- fold_tileop  ----

std::optional<Finding> fold_tileop(
    PlanOp const &op, std::size_t p, BindingTable &t, Ctx const &ctx)
{
    // Group order: shape, references, identity/linkage, topology,
    // flags, within-op distinctness — every row below in this order.

    // Row 1: shape — the three fields; access one of the three modes.
    for (Operand const &e : op.operands)
    {
        if (static_cast<std::uint8_t>(e.access) >
            static_cast<std::uint8_t>(Access::ReadWrite))
        {
            return at_queue("malformed-plan", p, op);
        }
    }
    // Row 2: the op's node of the inventory, then every operand's
    // (node, chunk) of the pools.
    if (!ctx.inv->node_valid(op.u.tile_op.node))
    {
        return at_queue("unknown-id", p, op);
    }
    for (Operand const &e : op.operands)
    {
        if (!ctx.inv->chunk_valid(e.node, e.chunk))
        {
            return at_queue("unknown-id", p, op);
        }
    }
    // Row 3: every operand's tile of the tile table.
    for (Operand const &e : op.operands)
    {
        if (!ctx.tiles->carries(e.tile))
        {
            return at_queue("unknown-tile", p, op);
        }
    }
    // Row 4: every operand's index in range.
    for (Operand const &e : op.operands)
    {
        if (e.index >= ctx.tiles->n(e.tile))
        {
            return at_queue("index-out-of-range", p, op);
        }
    }
    // Row 5: the executability table's row  — the one
    // placement fault class closed at check time.
    if (!ctx.runnable->can(op.u.tile_op.task, op.u.tile_op.node))
    {
        return at_queue("task-not-runnable", p, op);
    }
    // Row 6: the placement rule — one op, one node.
    for (Operand const &e : op.operands)
    {
        if (e.node != op.u.tile_op.node)
        {
            return at_queue("operand-node-mismatch", p, op);
        }
    }
    // Row 7: every operand linked, any access, and the binding it names
    // is THIS logical chunk. Without the identity half a plan could feed
    // another tile's bytes to the task and still pass.
    for (Operand const &e : op.operands)
    {
        if (!t.linked(e.node, e.chunk))
        {
            return at_queue("access-unlinked", p, op);
        }
        auto const held = t.at(e.node, e.chunk);
        if (!held.has_value()
            || !(held->first == LogicalChunk{e.tile, e.index}))
        {
            if (ctx.identity != nullptr)
            {
                ctx.identity->push_back({p, op});
            }
            break;
        }
    }
    // Row 8: every Read and ReadWrite operand actual — the
    // read-before-write guard.
    for (Operand const &e : op.operands)
    {
        if ((e.access == Access::Read ||
                e.access == Access::ReadWrite) &&
            !t.flag_actual(e.node, e.chunk))
        {
            return at_queue("read-outdated", p, op);
        }
    }
    // Row 9: no (node, chunk) twice — one full pass before row 10, the
    // table's own order (a chunk both read and written is one
    // ReadWrite entry, never two).
    {
        std::unordered_set<std::uint64_t> seen;
        for (Operand const &e : op.operands)
        {
            if (!seen.insert(chunk_key(e.node, e.chunk)).second)
            {
                return at_queue("duplicate-operand", p, op);
            }
        }
    }
    // Row 10: at most one Write/ReadWrite operand per logical chunk —
    // the single-writer guard (two writers could leave actual
    // replicas with different bytes).
    {
        std::unordered_set<std::uint64_t> writers;
        for (Operand const &e : op.operands)
        {
            if (e.access != Access::Read &&
                !writers.insert(logical_key(e.tile, e.index)).second)
            {
                return at_queue("duplicate-write", p, op);
            }
        }
    }
    // Effects, in order : every written operand ACTUAL
    // first, then every OTHER linked replica of each written logical
    // chunk OUTDATED — actual replicas included; Read operands change
    // no flag. Reading one replica while writing another is legal: the
    // read evaluated against the pre-op fold above, the write outdates
    // the read replica now.
    for (Operand const &e : op.operands)
    {
        if (e.access != Access::Read)
        {
            t.actualize(e.node, e.chunk);
        }
    }
    for (Operand const &e : op.operands)
    {
        if (e.access != Access::Read)
        {
            t.outdate_replicas(
                LogicalChunk{e.tile, e.index}, e.node, e.chunk);
        }
    }
    return std::nullopt;
}

// ---- the correspondence scan  ----

// One operand entry witnessed in the walk, kept with its op's queue
// position — the census names the offending op verbatim.
struct Observed
{
    std::size_t p;
    Operand entry;
};

// The per-occurrence accumulators are SoA over occurrences (see the
// Correspondence members): the access table is a CSR of (tile,
// access) rows — tile-sorted, first naming kept, binary-searched —
// the covered sets are flat (occurrence rows keyed by logical_key in
// one grow-at-the-tail array; the counts are never read at close,
// only membership), and the witnessed entries ride ONE flat array
// whose occurrence ranges are the runs' own boundaries (runs only
// append, so an occurrence's entries arrive contiguously — no
// per-occurrence container exists at any size). Task membership
// is a dense bitmap over the sequence's task ids. Same
// rules, same finding order, same verdicts — the hash-of-maps shape
// made the checker's cost grow badly at large single-shot plans; the
// SoA layout removes it.
class Correspondence
{
  public:
    Correspondence(std::vector<SeqTask> const &seq, TileTable const &tiles)
        : seq_(&seq), tiles_(&tiles)
    {
        // Task membership: one bitmap over the sequence's task ids.
        for (SeqTask const &st : seq)
        {
            max_task_ = std::max(max_task_, st.task);
        }
        std::size_t const bits =
            static_cast<std::size_t>(max_task_) + 1;
        of_seq_bits_.assign(bits / 64 + 1, 0);
        ran_bits_.assign(bits / 64 + 1, 0);
        // The access CSR: count each entry's FIRST naming per tile
        // (one pass into a tile-keyed dedup count via a per-entry
        // scratch — operands are few), then prefix, then fill. Rows
        // are tile-sorted; "first occurrence within THIS entry" keeps
        // the first access.
        acc_off_.assign(seq.size() + 1, 0);
        std::vector<TileAccess> row_scratch;
        for (std::size_t i = 0; i < seq.size(); ++i)
        {
            set_task_bit(of_seq_bits_, seq[i].task);
            row_scratch.assign(seq[i].operands.begin(),
                seq[i].operands.end());
            std::stable_sort(row_scratch.begin(), row_scratch.end(),
                [](TileAccess const &a, TileAccess const &b) {
                    return a.tile < b.tile;
                });
            std::size_t unique = 0;
            for (std::size_t k = 0; k < row_scratch.size(); ++k)
            {
                if (k != 0
                    && row_scratch[k].tile
                        == row_scratch[k - 1].tile)
                {
                    continue; // a later naming of the same tile keeps
                              // the FIRST access
                }
                ++unique;
            }
            acc_off_[i + 1] = acc_off_[i] + unique;
        }
        acc_tile_.resize(acc_off_.back());
        acc_access_.resize(acc_off_.back());
        for (std::size_t i = 0; i < seq.size(); ++i)
        {
            row_scratch.assign(seq[i].operands.begin(),
                seq[i].operands.end());
            std::stable_sort(row_scratch.begin(), row_scratch.end(),
                [](TileAccess const &a, TileAccess const &b) {
                    return a.tile < b.tile;
                });
            std::size_t at = acc_off_[i];
            for (std::size_t k = 0; k < row_scratch.size(); ++k)
            {
                if (k != 0
                    && row_scratch[k].tile
                        == row_scratch[k - 1].tile)
                {
                    continue;
                }
                acc_tile_[at] = row_scratch[k].tile;
                acc_access_[at] = row_scratch[k].access;
                ++at;
            }
        }
        // The covered sets' flat home: one array, per-occurrence
        // ranges carved as the runs witness entries (runs append, so
        // the current occurrence's keys grow at the tail).
        cov_key_.reserve(acc_off_.back());
        cov_off_.assign(seq.size() + 1, 0);
        entries_.reserve(acc_off_.back());
    }

    // A non-TILE_OP op at a queue position: only TILE_OP runs collapse,
    // so any other op breaks the run.
    void note_other() { prev_tileop_ = false; }

    // The task-invented rule fires during the walk — it is the
    // per-TILE_OP row, and it outranks every other correspondence
    // rule, so the earliest invented op is the finding wherever the
    // later rules might also complain. Findings accumulate here and
    // fire at close().
    std::optional<Finding> step(PlanOp const &op, std::size_t p)
    {
        TaskId const task = op.u.tile_op.task;
        if (task > max_task_ || !task_bit(of_seq_bits_, task))
        {
            return at_queue("task-invented", p, op);
        }
        // Run tracking: a maximal run of consecutive
        // TILE_OPs of one task collapses into one symbol — with the
        // ONE exception the engine's multiplied symbols make
        // mandatory: when the TASK SEQUENCE itself names the task
        // AGAIN at the very next entry, the consecutive TILE_OP is
        // that NEXT occurrence, not a continuation of the current
        // run. The planner pins one TILE_OP per task, so
        // two adjacent TILE_OPs of one task are always two submitted
        // instances ("a task the batch submits twice appears as two
        // symbols"), and without this split a no-pressure pool —
        // whose plans interleave nothing between adjacent TILE_OPs —
        // would merge the instances and mispair every symbol after
        // them (the >=7716-MiB ladder's deterministic census). A
        // genuine split (one occurrence, several TILE_OPs) still
        // merges: the sequence names the next entry's task, and the
        // run continues.
        bool const continuing = prev_tileop_ && prev_task_ == task;
        bool const sequence_repeats_now =
            runs_.size() < seq_->size() &&
            (*seq_)[runs_.size()].task == task;
        if (!continuing || sequence_repeats_now)
        {
            Run r;
            r.task = task;
            r.first_pos = p;
            r.ent_off = entries_.size();
            runs_.push_back(r);
            // The new occurrence's covered row STARTS at the flat
            // array's current tail (runs append, so occurrence rows
            // partition the array in arrival order; the row's END is
            // the NEXT occurrence's start, or the array's end at
            // close).
            std::size_t const occ = runs_.size() - 1;
            if (occ < seq_->size())
            {
                cov_off_[occ] = cov_key_.size();
            }
        }
        prev_tileop_ = true;
        prev_task_ = task;
        set_task_bit(ran_bits_, task); // the fact, per TASK: any
                                       // occurrence's run satisfies
                                       // the task

        // The one-for-one pairing: the i-th maximal run is
        // the i-th sequence entry's occurrence, so its entries
        // accumulate against THAT entry's operand list and accesses.
        // A run beyond the sequence's length accumulates nowhere —
        // the close check owns it, before it ever reads an
        // accumulator.
        std::size_t const occ = runs_.size() - 1;
        if (occ >= seq_->size())
        {
            return std::nullopt;
        }
        std::size_t const acc_lo = acc_off_[occ];
        std::size_t const acc_hi = acc_off_[occ + 1];
        std::size_t const cov_lo = cov_off_[occ];
        for (Operand const &e : op.operands)
        {
            entries_.push_back(Observed{p, e});
            // THIS entry's access for e.tile: the access row is
            // tile-sorted (binary search).
            std::size_t lo = acc_lo;
            std::size_t hi = acc_hi;
            bool access_matches = false;
            while (lo < hi)
            {
                std::size_t const mid = lo + (hi - lo) / 2;
                if (acc_tile_[mid] < e.tile)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }
            if (lo < acc_hi && acc_tile_[lo] == e.tile
                && acc_access_[lo] == e.access)
            {
                access_matches = true;
            }
            if (access_matches)
            {
                // An entry with this occurrence's access covers its
                // logical chunk; replica reads collapse to one
                // covered chunk. The covered row is sorted by
                // logical_key (the counts were never read at close —
                // membership only).
                std::uint64_t const key =
                    logical_key(e.tile, e.index);
                std::size_t clo = cov_lo;
                std::size_t chi = cov_key_.size();
                while (clo < chi)
                {
                    std::size_t const mid = clo + (chi - clo) / 2;
                    if (cov_key_[mid] < key)
                    {
                        clo = mid + 1;
                    }
                    else
                    {
                        chi = mid;
                    }
                }
                if (clo >= cov_key_.size()
                    || cov_key_[clo] != key)
                {
                    cov_key_.insert(
                        cov_key_.begin() + static_cast<std::ptrdiff_t>(clo),
                        key);
                }
            }
        }
        return std::nullopt;
    }

    // The end-of-queue rows, in order; the first violated
    // row is the finding.
    std::optional<Finding> close(std::vector<PlanOp> const &queue)
    {
        // Every TASK has at least one TILE_OP — any
        // occurrence's run satisfies the task; a repeat whose second
        // occurrence never ran is the strict prefix case below, not a
        // missing task.
        for (SeqTask const &st : *seq_)
        {
            if (!task_bit(ran_bits_, st.task))
            {
                return at_close("task-missing",
                    "task " + std::to_string(st.task) +
                        " has no TILE_OP");
            }
        }
        // The collapsed symbol sequence IS the task sequence
        // one-for-one. The first divergent run is the offender; extra
        // runs beyond the sequence diverge at their own start.
        for (std::size_t i = 0; i < runs_.size(); ++i)
        {
            if (i >= seq_->size() || runs_[i].task != (*seq_)[i].task)
            {
                return at_queue("task-order-violation",
                    runs_[i].first_pos, queue[runs_[i].first_pos]);
            }
        }
        if (runs_.size() < seq_->size())
        {
            // Only reachable when the sequence names one task twice:
            // every task appeared yet the symbols form
            // a strict prefix — the missing symbol never came.
            return at_close("task-order-violation",
                "the queue collapsed into " +
                    std::to_string(runs_.size()) +
                    " symbols, the task sequence holds " +
                    std::to_string(seq_->size()));
        }
        // Per occurrence, per operand tile, per index 0..n-1,
        // at least one operand entry names (tile, index) with THIS
        // occurrence's access — and none outside them is named
        // (coverage counts LOGICAL chunks). Coverage gaps first (the
        // sentence's own order), then outside naming. (After the
        // one-for-one pairing holds, every occurrence owns exactly one
        // run, so its covered row
        // [cov_off_[i], cov_off_[i+1]) and its witnessed-entry range
        // [runs_[i].ent_off, runs_[i+1].ent_off) are complete, in
        // arrival order — the same answers a per-occurrence container
        // would give.)
        for (std::size_t i = 0; i < seq_->size(); ++i)
        {
            SeqTask const &st = (*seq_)[i];
            std::size_t const cov_lo = cov_off_[i];
            std::size_t const cov_hi =
                i + 1 < seq_->size() ? cov_off_[i + 1]
                                     : cov_key_.size();
            for (TileAccess const &ta : st.operands)
            {
                ChunkIndex const n = tiles_->n(ta.tile);
                for (ChunkIndex idx = 0; idx < n; ++idx)
                {
                    std::uint64_t const key = logical_key(ta.tile, idx);
                    std::size_t lo = cov_lo;
                    std::size_t hi = cov_hi;
                    bool covered = false;
                    while (lo < hi)
                    {
                        std::size_t const mid = lo + (hi - lo) / 2;
                        if (cov_key_[mid] < key)
                        {
                            lo = mid + 1;
                        }
                        else
                        {
                            hi = mid;
                        }
                    }
                    if (lo < cov_hi && cov_key_[lo] == key)
                    {
                        covered = true;
                    }
                    if (!covered)
                    {
                        return at_close("operand-coverage-violation",
                            "tile " + std::to_string(ta.tile) +
                                " index " + std::to_string(idx) +
                                " of task " + std::to_string(st.task) +
                                " at sequence position " +
                                std::to_string(i) + " uncovered (n=" +
                                std::to_string(n) + ")",
                            &queue[runs_[i].first_pos],
                            runs_[i].first_pos);
                    }
                }
            }
            std::size_t const ent_hi =
                i + 1 < runs_.size() ? runs_[i + 1].ent_off
                                     : entries_.size();
            for (std::size_t k = runs_[i].ent_off; k < ent_hi; ++k)
            {
                Observed const &o = entries_[k];
                if (!access_row_has(i, o.entry.tile))
                {
                    return at_queue(
                        "operand-coverage-violation", o.p, queue[o.p]);
                }
            }
        }
        // Every operand entry's access IS its OWN occurrence's
        // access for its tile.
        for (std::size_t i = 0; i < seq_->size(); ++i)
        {
            std::size_t const ent_hi =
                i + 1 < runs_.size() ? runs_[i + 1].ent_off
                                     : entries_.size();
            for (std::size_t k = runs_[i].ent_off; k < ent_hi; ++k)
            {
                Observed const &o = entries_[k];
                std::size_t const lo = acc_off_[i];
                std::size_t const hi = acc_off_[i + 1];
                std::size_t a = lo;
                std::size_t b = hi;
                while (a < b)
                {
                    std::size_t const mid = a + (b - a) / 2;
                    if (acc_tile_[mid] < o.entry.tile)
                    {
                        a = mid + 1;
                    }
                    else
                    {
                        b = mid;
                    }
                }
                if (a < hi && acc_tile_[a] == o.entry.tile
                    && acc_access_[a] != o.entry.access)
                {
                    return at_queue("access-mode-mismatch", o.p,
                        queue[o.p]);
                }
            }
        }
        return std::nullopt;
    }

  private:
    struct Run
    {
        TaskId task;
        std::size_t first_pos; // the run's first TILE_OP's queue index
        std::size_t ent_off;   // its first witnessed entry's
                               // offset in the flat entries array
    };

    // The dense membership over the sequence's task ids
    // (max_task_ sized once; a task beyond it is never of the
    // sequence — answered without a probe).
    static void set_task_bit(std::vector<std::uint64_t> &bits, TaskId t)
    {
        bits[t >> 6] |= 1ULL << (t & 63);
    }
    static bool task_bit(
        std::vector<std::uint64_t> const &bits, TaskId t)
    {
        return (bits[t >> 6] >> (t & 63) & 1ULL) != 0;
    }

    // THIS occurrence's access row lookup: tile-sorted CSR row,
    // binary search.
    bool access_row_has(std::size_t occ, TileId tile) const
    {
        std::size_t lo = acc_off_[occ];
        std::size_t hi = acc_off_[occ + 1];
        while (lo < hi)
        {
            std::size_t const mid = lo + (hi - lo) / 2;
            if (acc_tile_[mid] < tile)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return lo < acc_off_[occ + 1] && acc_tile_[lo] == tile;
    }

    std::vector<SeqTask> const *seq_;
    TileTable const *tiles_;
    TaskId max_task_ = 0;
    std::vector<std::uint64_t> of_seq_bits_; // membership in the sequence
    std::vector<std::uint64_t> ran_bits_;    // any occurrence ran
    // The per-occurrence access rows (CSR): [acc_off_[i],
    // acc_off_[i+1]) over (acc_tile_, acc_access_), tile-sorted,
    // first naming kept.
    std::vector<std::size_t> acc_off_;
    std::vector<TileId> acc_tile_;
    std::vector<Access> acc_access_;
    // The covered sets: one flat key array partitioned by the runs'
    // own boundaries (cov_off_[i] written when occurrence i's run
    // starts; after the one-for-one pairing every row is complete).
    std::vector<std::size_t> cov_off_;
    std::vector<std::uint64_t> cov_key_;
    // The witnessed entries, flat, partitioned by the runs' ent_off.
    std::vector<Observed> entries_;
    std::vector<Run> runs_;
    bool prev_tileop_ = false;
    TaskId prev_task_ = 0;
};

// ---- the census render  ----

std::string access_char(Access a)
{
    switch (a)
    {
    case Access::Read: return "R";
    case Access::Write: return "W";
    case Access::ReadWrite: return "RW";
    }
    return "?";
}

std::string render_binding(EntryBinding const &b)
{
    return "(" + std::to_string(b.node) + "," + std::to_string(b.chunk) +
        ")=(" + std::to_string(b.tile) + "," + std::to_string(b.index) +
        "," + (b.actual ? "actual" : "outdated") + ")";
}

std::string render_finding(Finding const &f)
{
    std::string s = f.fault + " at ";
    switch (f.site)
    {
    case CheckSite::Queue:
        s += "op " + std::to_string(f.position);
        break;
    case CheckSite::Entry: s += "entry"; break;
    case CheckSite::Close: s += "close"; break;
    }
    if (!f.detail.empty())
    {
        s += " (" + f.detail + ")";
    }
    return s;
}

// The inputs' relevant facts: the cardinalities, the channel matrix
// bits, the tiles and runnable facts the plan and the sequence
// actually name, the task sequence, the plan's sequence number, and
// the chain echo when the pipeline supplied one — fault path only,
// never a product of a passing check.
std::string render_inputs(CheckInput const &in)
{
    Plan const &plan = *in.plan;
    NodeInventory const &inv = *in.inventory;

    std::string s = "nodes=" + std::to_string(inv.node_count()) +
        " cards=[";
    for (std::size_t n = 0; n < inv.node_count(); ++n)
    {
        if (n != 0)
        {
            s += " ";
        }
        s += std::to_string(
            inv.pool_cardinality(static_cast<NodeId>(n)));
    }
    s += "] channels=[";
    bool first = true;
    for (std::size_t a = 0; a < inv.node_count(); ++a)
    {
        for (std::size_t b = 0; b < inv.node_count(); ++b)
        {
            if (inv.has_channel(static_cast<NodeId>(a),
                    static_cast<NodeId>(b)))
            {
                if (!first)
                {
                    s += " ";
                }
                first = false;
                s += std::to_string(a) + "->" + std::to_string(b);
            }
        }
    }
    s += "]";

    // The tiles the plan and the sequence name, with the table's n (or
    // the missing-mark for a tile the table does not carry).
    std::unordered_set<TileId> named;
    for (EntryBinding const &b : plan.entry)
    {
        named.insert(b.tile);
    }
    for (PlanOp const &op : plan.queue)
    {
        switch (op.kind)
        {
        case PlanOpKind::Link: named.insert(op.u.link.tile); break;
        case PlanOpKind::Unlink: named.insert(op.u.unlink.tile); break;
        case PlanOpKind::Copy: named.insert(op.u.copy.tile); break;
        case PlanOpKind::TileOp:
            for (Operand const &e : op.operands)
            {
                named.insert(e.tile);
            }
            break;
        }
    }
    std::unordered_set<TaskId> tasks;
    for (SeqTask const &st : *in.task_sequence)
    {
        tasks.insert(st.task);
        for (TileAccess const &ta : st.operands)
        {
            named.insert(ta.tile);
        }
    }
    for (PlanOp const &op : plan.queue)
    {
        if (op.kind == PlanOpKind::TileOp)
        {
            tasks.insert(op.u.tile_op.task);
        }
    }
    s += " tiles={";
    first = true;
    for (TileId tile : named)
    {
        if (!first)
        {
            s += " ";
        }
        first = false;
        s += std::to_string(tile) + ":";
        s += in.tiles->carries(tile) ? std::to_string(in.tiles->n(tile))
                                     : "?";
    }
    s += "} runnable={";
    first = true;
    for (TaskId task : tasks)
    {
        if (!first)
        {
            s += " ";
        }
        first = false;
        s += std::to_string(task) + ":[";
        bool fnode = true;
        for (std::size_t n = 0; n < inv.node_count(); ++n)
        {
            if (in.runnable->can(task, static_cast<NodeId>(n)))
            {
                if (!fnode)
                {
                    s += " ";
                }
                fnode = false;
                s += std::to_string(n);
            }
        }
        s += "]";
    }
    s += "} sequence=[";
    first = true;
    for (SeqTask const &st : *in.task_sequence)
    {
        if (!first)
        {
            s += " ";
        }
        first = false;
        s += std::to_string(st.task) + ":(";
        bool fop = true;
        for (TileAccess const &ta : st.operands)
        {
            if (!fop)
            {
                s += " ";
            }
            fop = false;
            s += std::to_string(ta.tile) + ":" + access_char(ta.access);
        }
        s += ")";
    }
    s += "] plan_seq=" + std::to_string(plan.plan_seq);

    // The chain echo, when the pipeline supplied one — the differencing
    // reader sees both sides of boundary-discontinuity.
    if (in.chain.has_value())
    {
        s += " chain=[";
        first = true;
        for (EntryBinding const &b : *in.chain)
        {
            if (!first)
            {
                s += " ";
            }
            first = false;
            s += render_binding(b);
        }
        s += "]";
    }
    else
    {
        s += " chain=none";
    }
    return s;
}

// The verdict on a finding: the census assembled beside the plan,
// never written into it — the census travels beside the plan.
CheckResult fail(Finding f, BindingTable const &t, Plan const &plan,
    CheckInput const &in)
{
    CheckResult r;
    r.pass = false;
    r.fault = f.fault;
    r.site = f.site;
    r.position = f.position;
    r.op = f.op;

    Census c;
    c.fault = render_finding(f);
    c.op = f.op;
    c.fold = t.render(); // the fold snapshot, every node, free lists
                         // printed beside
    c.entry = plan.entry;
    // The ops prefix: everything folded before the offending op; the
    // whole queue for a close finding; none before op 0.
    if (f.site == CheckSite::Close)
    {
        c.prefix = plan.queue;
    }
    else if (f.site == CheckSite::Queue)
    {
        c.prefix.assign(plan.queue.begin(),
            plan.queue.begin() +
                static_cast<std::ptrdiff_t>(f.position));
    }
    c.inputs = render_inputs(in);
    r.census = std::move(c);
    return r;
}

} // namespace

// ---- check  ----

CheckResult check(CheckInput const &input)
{
    // The five mandatory inputs; a null among them is a caller defect,
    // loud, never a plan fault.
    require(input.plan, "plan");
    require(input.inventory, "inventory");
    require(input.tiles, "tiles");
    require(input.runnable, "runnable");
    require(input.task_sequence, "task_sequence");

    // Input coherence the fold presupposes: the sequence's operand
    // tiles are registry facts of the tile table (a miss is a registry
    // defect, loud — the fold's unknown-tile row owns the PLAN's
    // tiles, not the sequence's).
    for (SeqTask const &st : *input.task_sequence)
    {
        for (TileAccess const &ta : st.operands)
        {
            if (!input.tiles->carries(ta.tile))
            {
                throw std::runtime_error(
                    "plan_checker: the task sequence names tile " +
                    std::to_string(ta.tile) + " of task " +
                    std::to_string(st.task) +
                    ", which the tile table does not carry "
                    "");
            }
        }
    }

    Plan const &plan = *input.plan;
    std::vector<std::pair<std::size_t, PlanOp>> identity;
    Ctx const ctx{input.inventory, input.tiles, input.runnable, &identity};

    // T = BindingTable(inventory.cardinalities).
    std::vector<std::size_t> cards;
    cards.reserve(input.inventory->node_count());
    for (std::size_t n = 0; n < input.inventory->node_count(); ++n)
    {
        cards.push_back(
            input.inventory->pool_cardinality(static_cast<NodeId>(n)));
    }
    BindingTable t(cards);

    // The entry-state check, before op 0 : every entry
    // binding validated row by row, then folded in — T.link, and
    // T.actualize for an actual entry. The fold then preserves these
    // invariants by construction.
    for (std::size_t i = 0; i < plan.entry.size(); ++i)
    {
        EntryBinding const &b = plan.entry[i];
        std::string const verbatim =
            "binding " + std::to_string(i) + ": " + render_binding(b);
        if (!ctx.inv->chunk_valid(b.node, b.chunk))
        {
            return fail(
                at_entry("unknown-id", i, verbatim), t, plan, input);
        }
        if (!ctx.tiles->carries(b.tile))
        {
            return fail(
                at_entry("unknown-tile", i, verbatim), t, plan, input);
        }
        if (b.index >= ctx.tiles->n(b.tile))
        {
            return fail(at_entry("index-out-of-range", i, verbatim), t,
                plan, input);
        }
        LogicalChunk const lc{b.tile, b.index};
        if (t.binding_of(b.node, lc).has_value())
        {
            // At most one binding of one logical chunk per node.
            return fail(
                at_entry("double-binding", i, verbatim), t, plan,
                input);
        }
        if (t.linked(b.node, b.chunk))
        {
            // At most one binding per (node, chunk): two entry
            // bindings naming one chunk — different logical chunks,
            // else the row above fired — are double-link.
            return fail(
                at_entry("double-link", i, verbatim), t, plan, input);
        }
        t.link(b.node, b.chunk, lc);
        if (b.actual)
        {
            t.actualize(b.node, b.chunk);
        }
    }

    // The chaining check, whenever the pipeline supplies
    // the predecessor's exit fold: predicted-versus-predicted, over
    // ALL nodes' segments, binding for binding, flag for flag.
    if (input.chain.has_value() && t.snapshot() != *input.chain)
    {
        return fail(
            at_entry("boundary-discontinuity", 0,
                "the chain differs from the entry fold over all "
                "nodes"),
            t, plan, input);
    }

    // The fold: rows then effects per op, and the correspondence scan
    // over the same walk — a TILE_OP's fold rows run first, its
    // correspondence step second; any other op breaks the run
    // collapse.
    Correspondence corr(*input.task_sequence, *input.tiles);
    for (std::size_t p = 0; p < plan.queue.size(); ++p)
    {
        PlanOp const &op = plan.queue[p];
        std::optional<Finding> f;
        switch (op.kind)
        {
        case PlanOpKind::Link: f = fold_link(op, p, t, ctx); break;
        case PlanOpKind::Unlink: f = fold_unlink(op, p, t, ctx); break;
        case PlanOpKind::Copy: f = fold_copy(op, p, t, ctx); break;
        case PlanOpKind::TileOp: f = fold_tileop(op, p, t, ctx); break;
        default:
            // The queue contains EXACTLY the four kinds; any other
            // entry is malformed-plan.
            f = at_queue("malformed-plan", p, op);
            break;
        }
        if (f.has_value())
        {
            return fail(*f, t, plan, input);
        }
        if (op.kind == PlanOpKind::TileOp)
        {
            f = corr.step(op, p);
            if (f.has_value())
            {
                return fail(*f, t, plan, input);
            }
        }
        else
        {
            corr.note_other();
        }
    }
    if (auto f = corr.close(plan.queue))
    {
        return fail(*f, t, plan, input);
    }
    // Last row: a TILE_OP named a chunk bound to another logical chunk.
    // Reported only after the close correspondence, whose rows own the
    // more specific fault when both conditions hold.
    if (!identity.empty())
    {
        return fail(at_queue("identity-mismatch", identity.front().first,
                         identity.front().second),
            t, plan, input);
    }

    // Pass: the verdict and the exit state beside the plan — the plan
    // itself untouched, byte for byte.
    CheckResult r;
    r.pass = true;
    r.exit = t.snapshot();
    return r;
}

} // namespace nnhaul
