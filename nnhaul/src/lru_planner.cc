/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 * NNHaul is a task-based dynamic execution engine. This file is the
 * lowering walk's own acts: one pass over the batch's task sequence,
 * folding the four op kinds over a working copy of the binding table.
 *
 * @file src/lru_planner.cc
 * The walk behind src/lru_planner.hh:
 * a BindingTable working copy seeded at the entry state and advanced
 * by the SAME transitions as the ops appended — the exit state IS the
 * predicted fold. Per task, in submit order, per operand tile in the
 * order given, per chunk index ascending: ensure a binding on the
 * task's node with ACTUAL data by the TILE_OP's position — the replica
 * read when the node's binding is actual, a COPY from the lowest-NodeId
 * actual source when it is outdated or fresh (a host-staged two-hop
 * chain when no direct channel exists), a bare LINK for a Write (output
 * production is LINK plus Write). Eviction on an exhausted free list:
 * LRU by last planned touch — the victim-selection ORDER is fixed and
 * MUST NOT change (one policy, LRU) — with the data-survival rule
 * below. Pool exhaustion is the plan-time fault, censused at the state
 * read, never serviced. No scheduling: queue order IS task order; the
 * whole-tile lowering form is fixed.
 *
 * THE PINNED DETERMINISTIC ORDERS, stated once, are the planner's
 * whole freedom exercised:
 *
 * 1. Tasks in the request's order (the tile graph's topological
 *    order); operand tiles in the order given; chunk indexes 0..n-1
 *    ascending. Queue order IS this order.
 *
 * 2. THE WHOLE-TILE RULE: one TILE_OP per task, one
 *    operand entry per logical chunk of its tiles. The workload's
 *    kernels are whole-tile; chunk-parallel lowering is not built.
 *
 * 3. THE COPY SOURCE: the lowest NodeId holding an ACTUAL binding of
 *    the logical chunk with a channel to the task's node (devices
 *    ascending, the host last by its id k). Without a direct channel,
 *    the lowest-NodeId actual source staged through the HOST — the
 *    two-hop form: LINK(host) +
 *    COPY(src->host) + COPY(host->dst) + UNLINK(host), the residue
 *    UNLINK present. The host hop reuses the host's existing OUTDATED
 *    binding of the logical chunk when one is linked (no LINK, no
 *    residue UNLINK — the binding is not this walk's to drop); a
 *    fresh staging LINK takes pop_free(host) and censuses on empty.
 *
 * 4. THE LRU ORDER (the rule): a binding's recency IS the position of
 *    its last planned touch in the current plan build — every op that
 *    names a chunk touches it (LINK at creation, a COPY's src and dst,
 *    a TILE_OP's every operand), moving it to the order's back; UNLINK
 *    removes. Entry-state bindings carry no planned touch this build:
 *    they seed the order's FRONT (the eviction end) in ascending
 *    ChunkId per node — the oldest by construction, deterministically
 *    ordered among themselves. The victim is the front-most binding of
 *    the node EXCLUDING the current task's working set (the chunks
 *    touched while resolving the current task — the working-set guard:
 *    evicting what the TILE_OP is about to name is
 *    self-defeating and forbidden); when every linked chunk of the
 *    node is working set and the free list is empty, that is
 *    pool exhaustion, censused.
 *
 * 5. THE DATA-SURVIVAL RULE (this file's exact rule, combining the
 *    dying-victim discipline with the ENGINE's cross-batch liveness): a
 *    victim binding that is the LAST ACTUAL replica of
 *    its logical chunk is rescued — the bytes land on another node —
 *    UNLESS the tile is DEAD (the request's dead-tile set, the
 *    retirement riding this batch: its bindings drop at the tail
 *    anyway). A live tile's readers may live in LATER batches the
 *    planner cannot see, so its last actual replica never dies bare
 *    (a batch-scoped rule could lose a live tile's data; the
 *    engine's dead-tile set closes that gap). The
 *    rescue lands the data on the HOST first (the drain direction),
 *    then devices ascending, an existing OUTDATED binding of the
 *    logical chunk preferred over a fresh LINK — and only then does
 *    the victim UNLINK. No landing anywhere (no channel-bearing node
 *    with an outdated binding or a free chunk) is pool exhaustion,
 *    censused. An OUTDATED victim holds no data: it unlinks bare.
 *
 * 6. THE DYING-VICTIM FREE (release TIMING — liveness, never
 *    scheduling): every binding of a DEAD tile is unlinked BARE at the
 *    position after
 *    the tile's LAST USE of this batch (the queue head for a tile
 *    this batch never uses — the entry's dead residue of predecessor
 *    batches), so its chunks rejoin the free list BEFORE pressure can
 *    evict them and the retirement never squats on the pool mid-walk.
 *    No task after that position names the tile (last use is last by
 *    construction), so the fold stays legal and the bytes die exactly
 *    where the engine's trailing retirement would have dropped them
 *    anyway. This is also the ONE liveness class ahead of the LRU
 *    victim order (order 4): when eviction does fire, a binding of a
 *    dead tile with no remaining reader is taken FIRST, bare, before
 *    any live LRU victim is touched — the live victim order itself is
 *    LRU, unchanged.
 *
 * 7. THE EARLY REFETCH FIRE (the placement of the refetch COPY — the
 *    planner's authorship over its OWN auxiliary ops; the TASK order
 *    is untouched): a refetch COPY emitted at its consumer's resolution
 *    EXECUTES between the consumer's kernel and whatever last touched
 *    its destination chunk — on a saturated card the copy's latency is
 *    exposed on the critical path (the residual under pressure). The
 *    fire decouples the COPY's emission position from its consumer:
 *    a logical chunk the walk evicts (or the entry holds off-device)
 *    whose NEXT READING consumer lives later in this batch enters a
 *    PENDING queue keyed by that consumer's position; at every
 *    free-list gain (the dying frees' positions, the walk start) the
 *    queue drains through the SLACK GATE — the peak rule:
 *    fire only while the free list plus the releases the
 *    schedule GUARANTEES (the dying frees at their pinned positions)
 *    stays strictly above every intervening prefix's demands (fresh
 *    write-only namings, the pending queue itself, the current task's
 *    unmet needs) — so a fire never displaces a victim (an ungated
 *    fire spending the free list would force evictions that refetch
 *    what they displaced, a fire->evict->refire loop). A fired chunk
 *    enters PINNED — out of the recency
 *    order until its consumer claims it, so pressure between the fire
 *    and the consumer cannot throw the prefetch away; pins are spent
 *    only at pool exhaustion, the farthest consumer first, with the
 *    data-survival rule applied as for any victim. The fired LINK and
 *    COPY are ordinary ops at the drain's queue position; the event
 *    structure (the fold, the mint iff) derives them exactly as for a
 *    just-in-time refetch, and the copy overlaps the compute between
 *    the fire and its consumer.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "lru_planner.hh"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nnhaul
{

// ---- the checker's task sequence, from the request  ----

std::vector<SeqTask> to_seq_tasks(std::vector<PlannedTask> const &tasks)
{
    std::vector<SeqTask> seq;
    seq.reserve(tasks.size());
    for (PlannedTask const &t : tasks)
    {
        SeqTask st;
        st.task = t.task;
        st.operands.reserve(t.operands.size());
        for (TileUse const &u : t.operands)
        {
            st.operands.push_back(TileAccess{u.tile, u.access});
        }
        seq.push_back(std::move(st));
    }
    return seq;
}

namespace
{

    // The recency sentinel: a link field that is no chunk.
    constexpr ChunkId kNoChunk = std::numeric_limits<ChunkId>::max();

    // The dead-class victim scan's EXACT gate: a
    // per-node count of the recency list's members whose tile is DEAD
    // and READER-PASSED at the current task (the precise class the
    // LRU walk's first pass hunts — see lru_victim). Maintained at
    // the list's three mutation primitives and at the task boundary
    // that passes a dead tile's last reader:
    //   push_back — a (re)linked chunk enters counted iff its tile is
    //     dead and its tile's last read is already behind the current
    //     task (a never-read tile counts from task 0);
    //   list_remove — a counted member leaves, uncounted;
    //   the boundary flip — a dead tile whose last read was the task
    //     just finished turns reader-passed: its still-linked bindings
    //     (if any) become counted, probed through the live working
    //     table (the flip list is precomputed per plan; every dead
    //     tile flips at most once, so the whole gate costs O(dead
    //     tiles x chunks) per plan, against the O(list) scan it
    //     replaces per eviction).
    // EXACT, not an approximation: the count is zero exactly when no
    // in-list binding of the node is (dead, reader-passed) — the
    // first pass can return ONLY such a binding (the working-set
    // stamp guard further restricts, never widens), so a zero count
    // skips a pass that provably returns nothing: the victim choice,
    // plan content, op order, and every counter are identical.
    constexpr TileId kNoTileMark = std::numeric_limits<TileId>::max();

// The seed stamp: an entry binding's mark — never a task index of
// this build (task indexes are 0..|tasks|-1), so an untouched entry
// binding is evictable from task 0 on while a working-set chunk
// (stamped with the CURRENT task index) never is.
constexpr std::uint64_t kSeedStamp =
    std::numeric_limits<std::uint64_t>::max();

// The in-walk timers' clock (NNHAUL_PLAN_BREAKDOWN,
// steady nanoseconds; diagnosis only, never control flow).
inline std::uint64_t brk_now()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
struct BrkTimer
{
    bool on;
    std::uint64_t &acc;
    std::uint64_t t0;
    explicit BrkTimer(bool on_, std::uint64_t &acc_) noexcept :
        on(on_), acc(acc_), t0(on_ ? brk_now() : 0)
    {
    }
    ~BrkTimer() { acc += on ? brk_now() - t0 : 0; }
};
inline bool brk_on()
{
    static bool const on = env_flag_enabled("NNHAUL_PLAN_BREAKDOWN");
    return on;
}

// One intrusive recency link, addressed by the flattened slot
// base(node) + chunk :
// prev/next are the list's neighbors, in_list the membership. Free
// chunks sit out of the list; a binding leaves only at its UNLINK and
// re-enters at the back on its next touch.
struct RLink
{
    ChunkId prev = kNoChunk;
    ChunkId next = kNoChunk;
    bool in_list = false;
};

// The need under resolution — the census's "which task / tile chunk".
struct Need
{
    std::size_t task = 0;      // the batch position
    TaskId task_id = 0;
    TileId tile = 0;
    ChunkIndex index = 0;
    Access access = Access::Read;
    std::string what;          // the blocker's own words
};

// The walk's whole session: the state read's working copy, the
// per-node recency orders, the queue under construction, and the
// liveness model (the batch's remaining reads). Nothing persists
// across calls.
struct Walk
{
    NodeInventory const *inv = nullptr;
    TileTable const *tiles = nullptr;
    PlanRequest const *req = nullptr;

    std::vector<EntryBinding> entry; // sorted by (node, chunk)
    BindingTable working;            // the state read
    Plan plan;                       // under construction

    // The growth state: per node the WALK-START
    // cardinality (the plan's growth vector carries the FINAL
    // cardinality for every node this walk grew — new chunks are free
    // by construction, the executor commits them before the check).
    std::vector<std::size_t> growth_base_;
    std::vector<std::size_t> growth_final_;

    // Recency, flattened over (node, chunk) slots.
    std::vector<std::size_t> base_;
    std::vector<RLink> rlink_;
    std::vector<std::uint64_t> stamp_;
    std::vector<ChunkId> rfront_;
    std::vector<ChunkId> rback_;

    // The dead-class gate's state (the kNoTileMark block
    // explains how it is maintained): per slot the linked
    // chunk's tile and its counted-membership bit, per node the
    // count, and the precomputed per-boundary flip lists.
    std::vector<TileId> tile_of_;
    std::vector<std::uint8_t> deadcls_;
    std::vector<std::size_t> cls_count_;
    std::vector<std::vector<TileId>> flip_at_;

    // Dense read-only mirrors for the gate's hot test —
    // class_member_ runs at EVERY recency push (emit touch), so the
    // dead-tile bit and the last-read position live as arrays, sized
    // once per plan over the request's tile ids (registration numbers
    // tiles densely; the hash maps beside stay the source of truth).
    // Built BEFORE recency_seed — the seed's pushes consult them.
    std::vector<std::uint8_t> dead_bit_;
    std::vector<std::size_t> last_read_dense_;
    static constexpr std::size_t kNoReadPos =
        std::numeric_limits<std::size_t>::max();

    // The class test at the CURRENT task: dead tile, no remaining
    // reader (a never-read tile counts from task 0 — the dying free
    // of this batch's never-used dead tiles rides the queue head and
    // removes them from the list before any victim call).
    bool class_member_(TileId tile) const
    {
        if (tile >= dead_bit_.size() || dead_bit_[tile] == 0)
        {
            return false;
        }
        std::size_t const rd = last_read_dense_[tile];
        return rd == kNoReadPos ||
            rd < static_cast<std::size_t>(task_idx_);
    }

    // Liveness (the data-survival rule's reader model): per tile, the
    // LAST batch position that reads it (Read or ReadWrite). Absent:
    // no reader at all. DENSE is the source of truth (last_read_dense_,
    // below, kNoReadPos = absent): at a giant single-shot plan the
    // hash maps' per-plan build and per-push lookups became a large
    // cost, and the dense rows carry identical answers — the request's
    // tile ids are registration-dense, the Walk is per-plan, nothing
    // persists.
    // The dying-position model and the dead set ride the same rows
    // (last_use_dense_, dead_bit_).

    // Liveness (the dying-victim free's position model): per tile, the
    // LAST batch position that uses it with ANY access (a Write after
    // the last Read still needs the binding — output production over a
    // fresh LINK — so the free rides behind every use).
    std::vector<std::size_t> last_use_dense_; // kNoReadPos = absent

    // The dead set as an index (order 6's membership test is hot: the
    // victim scan consults it per candidate) — dense bit per tile id.
    // (dead_bit_, beside last_read_dense_ in the mirrors.)

    // The dying schedule (order 6): dying_at_[i] names the dead tiles
    // whose last use is batch position i-1 (dying_at_[0]: tiles this
    // batch never uses); the walk frees them right before task i.
    std::vector<std::vector<TileId>> dying_at_;

    // ---- the early refetch fire (pinned order 7) ----

    // One pending prefetch: a logical chunk with a READING consumer at
    // `next_read` (a Write-only naming never prefetches: output
    // production overwrites the bytes, a prefetch for it is a pure
    // round trip — write-only namings are skipped). The queue
    // stays sorted by (next_read, tile, index): the drain fires the
    // NEAREST consumer first.
    struct PendingFire
    {
        std::size_t next_read = 0;
        TileId tile = 0;
        ChunkIndex index = 0;
        bool operator<(PendingFire const &o) const
        {
            return next_read != o.next_read ? next_read < o.next_read
                                            : (tile != o.tile
                                                     ? tile < o.tile
                                                     : index < o.index);
        }
    };
    std::vector<PendingFire> pending_; // sorted; the drain's queue
    // (the queue's two mutation primitives — pending_insert_sorted /
    // pending_drop_front, carrying the gate trees' exact T update —
    // live beside the timers below, after MinTree)

    // The fired chunks (PINNED): per flattened slot, out of the recency
    // order until the consumer's claim, with the consumer's position
    // (the exhaustion spend takes the farthest first).
    std::vector<std::uint8_t> pinned_;
    std::vector<std::size_t> pin_consumer_;

    // The slack gate's books, once per plan (the born/retire prefix
    // arrays, per DEVICE node m, indexed by batch
    // position): born_prefix_[m][p+1] sums the whole chunk counts of
    // tiles named at position p by a WRITE-ONLY naming on node m (an
    // upper bound — a resident rewrite overcounts; a naming that also
    // reads claims through the pending queue's side instead);
    // retire_prefix_[m][p] sums the dying frees' whole chunk counts
    // landing before task p (the frees the schedule GUARANTEES — the
    // walk emits them at their pinned positions whatever else happens).
    std::vector<std::vector<std::size_t>> born_prefix_;
    std::vector<std::vector<std::size_t>> retire_prefix_;

    // The reader model's full schedule: per tile, the ascending batch
    // positions that READ it (Read or ReadWrite) — the pending queue's
    // next-use key. CSR over tile ids (rows
    // [uses_off_[t], uses_off_[t+1]) into
    // uses_pos_, ascending — identical rows to the per-tile vectors by
    // construction, one count pass and one fill pass, no scattered
    // blocks, nothing persists beyond the Walk).
    std::vector<std::size_t> uses_off_;  // tile + 1 -> row end
    std::vector<std::size_t> uses_pos_;  // the ascending positions

    // The walk's measurement counts (PlannedBatch::Stats).
    std::uint64_t evictions_ = 0;
    std::uint64_t rescues_ = 0;
    std::uint64_t refetches_ = 0;
    std::uint64_t prefetches_ = 0;
    std::uint64_t dying_frees_ = 0;
    std::uint64_t pin_spends_ = 0; // exhaustion pin spends

    // The fragmentation meter (diagnosis only): multi-chunk
    // contiguous-handout misses — how many, the largest chain that
    // missed, and the longest free run seen at any miss.
    std::uint64_t run_misses_ = 0;
    ChunkIndex run_miss_want_ = 0;
    std::size_t run_miss_longest_ = 0;

    // In-walk cumulative timers (NNHAUL_PLAN_BREAKDOWN
    // only): the walk's own cost centers, nested ones included in
    // their parents (evict includes rescue+enqueue; drain includes
    // gate+pop_free_run+emit). Diagnosis only, never control flow.
    mutable std::uint64_t ns_victim_ = 0;
    mutable std::uint64_t ns_spend_ = 0;
    mutable std::uint64_t ns_evict_ = 0;
    mutable std::uint64_t ns_rescue_ = 0;
    mutable std::uint64_t ns_enq_ = 0;
    mutable std::uint64_t ns_drain_ = 0;
    mutable std::uint64_t ns_gate_ = 0;
    mutable std::uint64_t ns_pfr_ = 0; // pop_free_run call sites
    mutable std::uint64_t ns_emit_ = 0; // emit_link/copy/unlink/tile_op
    mutable std::uint64_t ns_fetch_ = 0;
    mutable std::uint64_t n_victim_ = 0;
    mutable std::uint64_t n_vscan_ = 0; // victim calls that ran the
                                        // dead-class pass (the gate
                                        // let them through)
    mutable std::uint64_t n_gate_ = 0;
    mutable std::uint64_t n_pfr_ = 0;

    // The slack gate's EXACT incremental form. The scan's
    // running value decomposes over the batch position k as
    //
    //   running(k) = base + T[k],   base = free_eff - retire[t1]
    //                                   + born[t1]        (per call),
    //   T[k] = retire[k] - born[k+1] - pending_due(<=k),
    //
    // with retire/born the PLAN-STATIC prefix books and
    // pending_due(<=k) = |{e in pending_ : e.next_read <= k}|. T
    // changes at exactly TWO primitives: an enqueue of next_read q
    // subtracts 1 from T[k] for every k >= q, and the drain's front
    // erase of q adds 1 back — a range-add on a suffix. One lazy
    // range-add/range-min segment tree per node m (over the node's
    // own static D_m[k] = retire[m][k] - born[m][k+1], lazily built
    // with the pending_dues of the moment) answers the gate's ONE
    // question — min over k in [t1, horizon] of T[k] — in O(log n),
    // and the verdict is the ORIGINAL predicate EXACTLY, both
    // directions: all running > 0 iff base + min > 0. No
    // approximation, no one-sided bound: the tree's T equals the
    // scan's T at every k by construction, so the fired/dropped/break
    // decisions — and through them the plan content, op order, and
    // every counter — are identical. (The mass it replaces: repeated
    // full rescans on every steady plan, each re-scanning the whole
    // horizon prefix; the tree answers each in a dozen node visits.)
    struct MinTree
    {
        std::size_t n = 0;
        std::vector<std::int64_t> mn, lz;
        void build(std::vector<std::int64_t> const &vals)
        {
            n = vals.size();
            mn.assign(4 * (n ? n : 1), 0);
            lz.assign(4 * (n ? n : 1), 0);
            if (n != 0)
            {
                build_(1, 0, n - 1, vals);
            }
        }
        void build_(std::size_t v, std::size_t l, std::size_t r,
            std::vector<std::int64_t> const &vals)
        {
            if (l == r)
            {
                mn[v] = vals[l];
                return;
            }
            std::size_t const mid = l + (r - l) / 2;
            build_(2 * v, l, mid, vals);
            build_(2 * v + 1, mid + 1, r, vals);
            mn[v] = std::min(mn[2 * v], mn[2 * v + 1]);
        }
        void push(std::size_t v)
        {
            if (lz[v] != 0)
            {
                for (std::size_t c = 2 * v; c <= 2 * v + 1; ++c)
                {
                    mn[c] += lz[v];
                    lz[c] += lz[v];
                }
                lz[v] = 0;
            }
        }
        void add(std::size_t l, std::size_t r, std::int64_t d)
        {
            if (n != 0)
            {
                add_(1, 0, n - 1, l, r, d);
            }
        }
        void add_(std::size_t v, std::size_t l, std::size_t r,
            std::size_t pl, std::size_t pr, std::int64_t d)
        {
            if (pr < l || r < pl)
            {
                return;
            }
            if (pl <= l && r <= pr)
            {
                mn[v] += d;
                lz[v] += d;
                return;
            }
            push(v);
            std::size_t const mid = l + (r - l) / 2;
            add_(2 * v, l, mid, pl, pr, d);
            add_(2 * v + 1, mid + 1, r, pl, pr, d);
            mn[v] = std::min(mn[2 * v], mn[2 * v + 1]);
        }
        std::int64_t min(std::size_t l, std::size_t r)
        {
            return n != 0 ? min_(1, 0, n - 1, l, r)
                          : std::numeric_limits<std::int64_t>::max();
        }
        std::int64_t min_(std::size_t v, std::size_t l, std::size_t r,
            std::size_t ql, std::size_t qr)
        {
            if (qr < l || r < ql)
            {
                return std::numeric_limits<std::int64_t>::max();
            }
            if (ql <= l && r <= qr)
            {
                return mn[v];
            }
            push(v);
            std::size_t const mid = l + (r - l) / 2;
            return std::min(min_(2 * v, l, mid, ql, qr),
                min_(2 * v + 1, mid + 1, r, ql, qr));
        }
    };
    std::vector<MinTree> gate_trees_;     // per node, over T[k]
    std::vector<std::uint8_t> gate_built_; // lazily, at first gate
    std::size_t gate_n_ = 0;               // req->tasks.size() + 1

    // The pending queue's two primitives, carrying T's exact update.
    void pending_drop_front()
    {
        std::size_t const q = pending_.front().next_read;
        pending_.erase(pending_.begin());
        for (std::size_t m = 0; m < gate_trees_.size(); ++m)
        {
            if (gate_built_[m] != 0)
            {
                gate_trees_[m].add(q, gate_n_ - 1, +1);
            }
        }
    }
    void pending_insert_sorted(PendingFire entry)
    {
        auto it = std::lower_bound(
            pending_.begin(), pending_.end(), entry);
        pending_.insert(it, entry);
        for (std::size_t m = 0; m < gate_trees_.size(); ++m)
        {
            if (gate_built_[m] != 0)
            {
                gate_trees_[m].add(entry.next_read, gate_n_ - 1, -1);
            }
        }
    }

    std::size_t task_idx_ = 0; // the working-set stamp domain
    bool task_served_ = false; // the current task's allocations done
    Need need_;                // the census's need, kept current

    // ---- slot addressing ----

    std::size_t slot(NodeId n, ChunkId c) const
    {
        return base_[n] + c;
    }

    // ---- the recency order (pinned order 4) ----

    void recency_seed()
    {
        // Entry bindings, ascending ChunkId per node, pushed to the
        // back in that order: the front (the eviction end) is the
        // node's lowest entry ChunkId, and every touched chunk lands
        // behind them all.
        for (EntryBinding const &b : entry)
        {
            push_back(b.node, b.chunk, kSeedStamp);
        }
    }

    void push_back(NodeId n, ChunkId c, std::uint64_t stamp)
    {
        std::size_t const s = slot(n, c);
        rlink_[s] = RLink{rback_[n], kNoChunk, true};
        if (rback_[n] == kNoChunk)
        {
            rfront_[n] = c;
        }
        else
        {
            rlink_[slot(n, rback_[n])].next = c;
        }
        rback_[n] = c;
        stamp_[s] = stamp;
        // The dead-class gate's entry bookkeeping (see the
        // kNoTileMark block): the freshly (re)linked chunk counts iff
        // its tile is dead and reader-passed at the current task. The
        // working table already carries the binding at every caller
        // (emit_link links first, then touches; recency_seed runs
        // after the entry links), so the tile read is valid here.
        TileId tile = kNoTileMark;
        auto const held = working.at(n, c);
        if (held.has_value())
        {
            tile = held->first.tile;
        }
        tile_of_[s] = tile;
        deadcls_[s] = tile != kNoTileMark && class_member_(tile) ? 1 : 0;
        if (deadcls_[s] != 0)
        {
            ++cls_count_[n];
        }
    }

    void list_remove(NodeId n, ChunkId c)
    {
        std::size_t const s = slot(n, c);
        if (!rlink_[s].in_list)
        {
            return;
        }
        if (deadcls_[s] != 0)
        {
            --cls_count_[n]; // the member leaves with the link
            deadcls_[s] = 0;
        }
        if (rlink_[s].prev != kNoChunk)
        {
            rlink_[slot(n, rlink_[s].prev)].next = rlink_[s].next;
        }
        else
        {
            rfront_[n] = rlink_[s].next;
        }
        if (rlink_[s].next != kNoChunk)
        {
            rlink_[slot(n, rlink_[s].next)].prev = rlink_[s].prev;
        }
        else
        {
            rback_[n] = rlink_[s].prev;
        }
        rlink_[s] = RLink{};
    }

    // One planned touch: to the back, stamped with the current task.
    void touch(NodeId n, ChunkId c)
    {
        std::size_t const s = slot(n, c);
        if (rlink_[s].in_list)
        {
            list_remove(n, c);
        }
        push_back(n, c, static_cast<std::uint64_t>(task_idx_));
    }

    // The LRU victim: the front-most binding of node n whose stamp is
    // not the current task's (the working-set guard). Nullopt when
    // every linked chunk of n is working set — pool-exhaustion's case.
    //
    // Order 6's one liveness class AHEAD of the recency order: a
    // binding of a DEAD tile with no remaining reader is taken FIRST,
    // bare (the dying-victim discipline) — the walk's dying frees make
    // the class empty in the common case, and the scan keeps it exact
    // for the shapes they cannot cover (a dead tile whose last use is
    // still ahead, freed there, but whose chunk is wanted NOW: it has
    // a remaining reader and stays a live-shape victim — the class is
    // precisely dead AND reader-passed). The LIVE victim order is LRU,
    // unchanged.
    //
    // The first pass below is O(recency list) per eviction: a
    // scattered walk with hash probes that, at steady load, never finds
    // a member — the dying frees keep the class empty, and
    // rescues == evictions is the witness. The per-node
    // cls_count_ (maintained as the kNoTileMark block describes) gates
    // it exactly: a zero count proves the pass would return nothing, so
    // the walk
    // goes straight to the live LRU pass — the victim choice, plan
    // content, op order, and every counter are identical either way.
    std::optional<ChunkId> lru_victim(NodeId n) const
    {
        BrkTimer const brk(brk_on(), ns_victim_);
        ++n_victim_;
        if (cls_count_[n] != 0)
        {
            ++n_vscan_;
            for (ChunkId c = rfront_[n]; c != kNoChunk;)
            {
                std::size_t const s = slot(n, c);
                if (stamp_[s] != static_cast<std::uint64_t>(task_idx_))
                {
                    auto const held = working.at(n, c);
                    if (held.has_value() && tile_is_dead(held->first.tile)
                        && !has_remaining_reader(held->first.tile))
                    {
                        return c; // the dying class: bare, first
                    }
                }
                c = rlink_[s].next;
            }
        }
        for (ChunkId c = rfront_[n]; c != kNoChunk;)
        {
            std::size_t const s = slot(n, c);
            if (stamp_[s] != static_cast<std::uint64_t>(task_idx_))
            {
                return c;
            }
            c = rlink_[s].next;
        }
        return std::nullopt;
    }

    // ---- the early refetch fire (pinned order 7) ----

    void clear_pin(NodeId n, ChunkId c)
    {
        std::size_t const s = slot(n, c);
        pinned_[s] = 0;
        pin_consumer_[s] = 0;
    }

    // A fired chunk leaves the recency order until its consumer claims
    // it: no eviction between the fire and the claim can take it (the
    // pin is the ungated hoist's missing half — a fired chunk IN the
    // order would be its farthest-future front, pressure's first
    // victim, and the prefetch would die exactly when needed).
    void pin_chunk(NodeId n, ChunkId c, std::size_t consumer)
    {
        std::size_t const s = slot(n, c);
        list_remove(n, c);
        pinned_[s] = 1;
        pin_consumer_[s] = consumer;
    }

    // The pending queue's key: the next batch position that READS the
    // tile chunk after `from` (a Write-only naming is not a prefetch
    // consumer). Nullopt when no reading consumer remains. The
    // CSR row scan gives the same rows as a per-tile vector walk.
    std::optional<std::size_t> next_read_after(
        TileId tile, std::size_t from) const
    {
        if (static_cast<std::size_t>(tile) + 1 >= uses_off_.size())
        {
            return std::nullopt;
        }
        for (std::size_t k = uses_off_[tile];
             k < uses_off_[tile + 1]; ++k)
        {
            std::size_t const pos = uses_pos_[k];
            if (pos > from)
            {
                return pos;
            }
        }
        return std::nullopt;
    }

    // Enqueue one pending fire for a logical chunk the walk is evicting
    // (or the entry holds off the reader's node) whose bytes survive
    // somewhere: only a READING consumer makes the bytes worth moving
    // early. Deduped: a second enqueue while the first entry stands
    // finds the OLD next use (the entry is replaced only once gone).
    void enqueue_pending(TileId tile, ChunkIndex index)
    {
        BrkTimer const brk(brk_on(), ns_enq_);
        std::optional<std::size_t> const next =
            next_read_after(tile, task_idx_);
        if (!next.has_value())
        {
            return; // no reading consumer ahead: nothing to fire for.
        }
        PendingFire entry{*next, tile, index};
        for (auto scan = pending_.begin(); scan != pending_.end(); ++scan)
        {
            if (scan->tile == tile && scan->index == index)
            {
                return; // already queued.
            }
        }
        pending_insert_sorted(entry);
    }

    // The direct actual source for a fire: the lowest NodeId holding an
    // ACTUAL binding of the logical chunk with a channel to n — the
    // same choice fetch_actual makes at the consumer (devices
    // ascending, the host last by its id).
    std::optional<ChunkLoc> direct_actual_source(
        LogicalChunk lc, NodeId n) const
    {
        for (std::size_t s = 0; s < inv->node_count(); ++s)
        {
            NodeId const src = static_cast<NodeId>(s);
            if (src == n)
            {
                continue;
            }
            std::optional<ChunkId> sb = working.binding_of(src, lc);
            if (sb.has_value() && working.flag_actual(src, *sb) &&
                inv->has_channel(src, n))
            {
                return ChunkLoc{src, *sb};
            }
        }
        return std::nullopt;
    }

    // THE SLACK GATE (the peak rule): fire for a consumer at `horizon`
    // on node m only while,
    // for EVERY prefix k in (task_idx_, horizon], the free list plus
    // the guaranteed releases stays strictly above the demands:
    //
    //   free_eff + (retire[k] - retire[t+1])
    //            > (born[k+1] - born[t+1]) + pending_due(<=k)
    //
    // free_eff discounts the current task's unmet allocations (the
    // drain never runs mid-allocation, so it is 0 beside at the walk
    // start); retire counts only the dying frees the SCHEDULE
    // guarantees at their pinned positions; born counts fresh
    // write-only namings (an upper bound, always gate-closing);
    // pending_due counts the queue's own claims, the firing head among
    // them — a fire consumes one free chunk and removes one pending
    // claim, so the inequality stands for the next head of the same
    // horizon. A fire that would dip any prefix to the last slot would
    // force an eviction before the consumer — the displaced victim's
    // own refetch a round trip the prefetch only traded for (an
    // ungated fire's failure) — so a saturated moment fires
    // nothing and the just-in-time path covers.
    bool gate_open(NodeId m, std::size_t horizon)
    {
        BrkTimer const brk(brk_on(), ns_gate_);
        ++n_gate_;
        std::vector<std::size_t> const &retire = retire_prefix_[m];
        std::vector<std::size_t> const &born = born_prefix_[m];
        std::int64_t free_eff = static_cast<std::int64_t>(
            working.free_count(m));
        if (!task_served_ && task_idx_ < req->tasks.size())
        {
            // The walk-start drain stands before task 0's own
            // allocations: discount them (dedup tiles, count unbound
            // chunk indexes on m).
            std::vector<TileId> seen;
            for (TileUse const &u : req->tasks[task_idx_].operands)
            {
                if (std::find(seen.begin(), seen.end(), u.tile) !=
                    seen.end())
                {
                    continue;
                }
                seen.push_back(u.tile);
                ChunkIndex const n = tiles->n(u.tile);
                for (ChunkIndex ix = 0; ix < n; ++ix)
                {
                    if (!working.binding_of(
                            m, LogicalChunk{u.tile, ix})
                             .has_value())
                    {
                        --free_eff;
                    }
                }
            }
        }
        std::size_t const t1 = task_idx_ + 1;
        // The exact incremental verdict (the MinTree block's
        // algebra): the scan's every running(k) over the range
        // [t1, horizon] equals base + T[k] with base and T defined
        // there, the tree holds T over every k of the plan, and the
        // boolean "every running > 0" is "base + min T > 0" — the
        // same predicate on the same integers, no approximation.
        if (gate_built_[m] == 0)
        {
            // Lazily build the node's tree: its static D plus the
            // pending_dues of THIS moment (the queue's two
            // primitives have kept every already-built tree exact;
            // an unbuilt one absorbs the current state at birth).
            std::vector<std::int64_t> t(gate_n_, 0);
            std::size_t due = 0;
            for (std::size_t k = 0; k < gate_n_; ++k)
            {
                while (due < pending_.size() &&
                    pending_[due].next_read <= k)
                {
                    ++due;
                }
                t[k] = static_cast<std::int64_t>(retire[k]) -
                    static_cast<std::int64_t>(born[k + 1]) -
                    static_cast<std::int64_t>(due);
            }
            gate_trees_[m].build(t);
            gate_built_[m] = 1;
        }
        std::int64_t const base = free_eff -
            static_cast<std::int64_t>(retire[t1]) +
            static_cast<std::int64_t>(born[t1]);
        std::int64_t const mink = gate_trees_[m].min(t1, horizon);
        return base + mink > 0;
    }

    // The drain: fire pending prefetches a free-list gain can spare.
    // Never called from inside an allocation (the freed chunk an
    // in-flight acquire is about to take is owed to its caller): the
    // drain runs at the walk start and at task boundaries, after the
    // dying frees landed. Entries whose consumer has arrived, whose
    // reader node already holds a binding, or whose bytes exist
    // nowhere actual drop out — the consumer covers itself just in
    // time, exactly as it always did.
    void drain_pending()
    {
        BrkTimer const brk(brk_on(), ns_drain_);
        while (!pending_.empty())
        {
            PendingFire const head = pending_.front();
            if (head.next_read <= task_idx_)
            {
                pending_drop_front();
                continue;
            }
            NodeId const m = req->tasks[head.next_read].node;
            if (inv->is_host(m))
            {
                // A host reader reads the host binding in place; its
                // resolution owns any fetch it still needs.
                pending_drop_front();
                continue;
            }
            LogicalChunk const lc{head.tile, head.index};
            if (working.binding_of(m, lc).has_value())
            {
                pending_drop_front();
                continue;
            }
            std::optional<ChunkLoc> const src =
                direct_actual_source(lc, m);
            if (!src.has_value())
            {
                pending_drop_front();
                continue;
            }
            if (!gate_open(m, head.next_read))
            {
                break; // the nearest consumer's books shut: stop.
            }
            // The fired destination: the tile's whole CHAIN RUN when
            // the free list carries one (pop_free_run(n) — the entry
            // lands at base+index, its siblings' entries find their
            // slots free, and the consumer's pass continues the chain
            // through acquire_for's reservation), else the LOWEST free
            // chunk — a RUN EDGE by construction. Never the blind LIFO
            // back: the fires land by the dozen in one drain, and a
            // back pop shreds the interiors of the long free runs the
            // multi-chunk chains depend on (the staged-gemm cost is
            // the price of the shredding
            // — large workspace staging under pressure).
            //
            // A rejected alternative, kept out deliberately: a chain
            // continuation across one tile's fires (seed the base at
            // the first fired chunk, later chunks prefer base+index)
            // would land multi-chunk chains contiguous — but ANY
            // device-landing change is ENTANGLED with the walk's own
            // probes: resolve()'s continuation tests free(b+ix) of a
            // partially-resident chain, and the drain's own
            // free(m, want) — both read the free set that every prior
            // landing shaped, so the acquire fallbacks fire at
            // different moments and the VICTIM SET drifts — different
            // tiles copied, same op counts. The host side carries no
            // such probe (no host tasks, no host pins, nothing scans
            // host runs) and changes nothing; see rescue's note.
            ChunkIndex const chain = tiles->n(lc.tile);
            std::optional<ChunkId> c;
            if (chain > 1)
            {
                std::optional<ChunkId> full;
                {
                    BrkTimer const brk(brk_on(), ns_pfr_);
                    ++n_pfr_;
                    full = working.pop_free_run(m, chain);
                }
                if (full.has_value())
                {
                    ChunkId const want = static_cast<ChunkId>(
                        *full + head.index);
                    if (want < *full + chain && working.free(m, want))
                    {
                        c = want; // the chain's own slot, index-aligned
                    }
                }
            }
            if (!c.has_value())
            {
                BrkTimer const brk(brk_on(), ns_pfr_);
                ++n_pfr_;
                c = working.pop_free_run(m, 1); // the edge fallback
            }
            if (!c.has_value())
            {
                break; // no free chunk now; a later gain retries.
            }
            pending_drop_front();
            emit_link(lc.tile, lc.index, m, *c);
            emit_copy(lc.tile, lc.index, *src, ChunkLoc{m, *c});
            pin_chunk(m, *c, head.next_read);
            ++prefetches_;
        }
    }

    // The exhaustion spend: when the
    // free list is empty and every linked chunk is the current task's
    // working set, a PINNED chunk is the remaining capacity — spend the
    // one whose consumer is FARTHEST away (every fired chunk's bytes
    // also live at its source, so the cost is the same whichever is
    // taken; keep the prefetches this batch reads sooner). The spent
    // chunk re-enters the pending queue for its still-ahead consumer.
    // Returns true when a pin was spent.
    bool spend_pinned(NodeId n)
    {
        BrkTimer const brk(brk_on(), ns_spend_);
        std::optional<ChunkId> best;
        std::size_t best_consumer = 0;
        ChunkId const card = static_cast<ChunkId>(
            inv->pool_cardinality(n));
        for (ChunkId c = 0; c < card; ++c)
        {
            std::size_t const s = slot(n, c);
            if (pinned_[s] == 0)
            {
                continue;
            }
            if (!best.has_value() || pin_consumer_[s] > best_consumer)
            {
                best = c;
                best_consumer = pin_consumer_[s];
            }
        }
        if (!best.has_value())
        {
            return false;
        }
        ++pin_spends_; // the meter: a pinned chunk was the capacity
        clear_pin(n, *best);
        evict_one(n, *best); // the survival rule, then the unlink
        // (evict_one's own enqueue re-queues the spent prefetch for
        // its still-ahead consumer.)
        return true;
    }

    // ---- the emissions: op, fold, touch — always together ----

    void emit_link(TileId tile, ChunkIndex index, NodeId n, ChunkId c)
    {
        BrkTimer const brk(brk_on(), ns_emit_);
        plan.queue.push_back(make_link(tile, index, n, c));
        working.link(n, c, LogicalChunk{tile, index});
        touch(n, c); // starts outdated
    }

    void emit_unlink(TileId tile, ChunkIndex index, NodeId n, ChunkId c)
    {
        BrkTimer const brk(brk_on(), ns_emit_);
        plan.queue.push_back(make_unlink(tile, index, n, c));
        working.unlink(n, c);
        list_remove(n, c); // the recency ends at the unlink
        clear_pin(n, c);   // a pinned chunk's pin ends with the binding
    }

    void emit_copy(TileId tile, ChunkIndex index, ChunkLoc src,
        ChunkLoc dst)
    {
        BrkTimer const brk(brk_on(), ns_emit_);
        plan.queue.push_back(make_copy(tile, index, src, dst));
        working.actualize(dst.node, dst.chunk); // dst becomes actual
        touch(src.node, src.chunk);
        touch(dst.node, dst.chunk);
    }

    void emit_tile_op(TaskId task, NodeId n, std::vector<Operand> ops)
    {
        BrkTimer const brk(brk_on(), ns_emit_);
        plan.queue.push_back(make_tile_op(task, n, std::move(ops)));
        // The fold's effects, mirroring the checker's fold_tileop
        // exactly : every written operand ACTUAL first, then
        // every OTHER linked replica of each written logical chunk
        // OUTDATED — actual replicas included.
        std::vector<Operand> const &held = plan.queue.back().operands;
        for (Operand const &e : held)
        {
            if (e.access != Access::Read)
            {
                working.actualize(e.node, e.chunk);
            }
        }
        for (Operand const &e : held)
        {
            if (e.access != Access::Read)
            {
                working.outdate_replicas(
                    LogicalChunk{e.tile, e.index}, e.node, e.chunk);
            }
        }
        for (Operand const &e : held)
        {
            touch(e.node, e.chunk); // the op names it: a planned touch
        }
    }

    // ---- the state read's queries  ----

    // The actual replicas of a logical chunk, excluding node skip.
    //! Session scratch: the walk
    ///  calls these per eviction/rescue probe — no per-call allocation
    ///  (avoids the malloc churn of a fresh vector per probe).
    mutable std::vector<ChunkLoc> scratch_replicas_;
    std::vector<ChunkLoc> const &actual_replicas(LogicalChunk lc,
        NodeId skip) const
    {
        std::vector<ChunkLoc> &out = scratch_replicas_;
        out.clear();
        for (std::size_t m = 0; m < inv->node_count(); ++m)
        {
            if (m == skip)
            {
                continue;
            }
            auto b = working.binding_of(static_cast<NodeId>(m), lc);
            if (b.has_value() &&
                working.flag_actual(static_cast<NodeId>(m), *b))
            {
                out.push_back(
                    ChunkLoc{static_cast<NodeId>(m), *b});
            }
        }
        return out;
    }

    // The reader model: a tile's data must survive at and after the
    // current task iff some task at this position or later reads it.
    // Dense row (a tile out of the row has no reader).
    bool has_remaining_reader(TileId tile) const
    {
        if (tile >= last_read_dense_.size())
        {
            return false;
        }
        std::size_t const rd = last_read_dense_[tile];
        return rd != kNoReadPos && rd >= static_cast<std::size_t>(task_idx_);
    }

    // The ENGINE's liveness half (cross-batch survival): a tile NOT in
    // the request's dead set is LIVE —
    // its readers may live in LATER batches the planner cannot see —
    // so its last actual replica is rescued whatever this batch's own
    // reader model says. A dead tile's bindings drop at their dying
    // positions (order 6; the riding retirement); its bytes need not
    // survive.
    bool tile_is_dead(TileId tile) const
    {
        return tile < dead_bit_.size() && dead_bit_[tile] != 0;
    }

    // ---- the loud census  ----

    [[noreturn]] void census_pool_exhaustion(NodeId n)
    {
        Census c;
        c.fault = "pool-exhaustion";
        c.op = std::nullopt; // the state read faults, not an op
        c.fold = working.render(); // every node, free lists beside
        c.applied.clear();         // plan time: no applied table
        c.entry = entry;
        c.prefix = plan.queue;     // the ops prefix so far
        c.issue_prefix.clear();    // plan time: nothing issued
        c.inputs = render_inputs(n);
        throw PlanCensusError(
            "pool-exhaustion: node " + std::to_string(n) + " at task " +
                std::to_string(need_.task) + " (tid " +
                std::to_string(need_.task_id) + "), operand tile " +
                std::to_string(need_.tile) + " chunk " +
                std::to_string(need_.index) + " — " + need_.what,
            std::move(c));
    }

    std::string render_inputs(NodeId n) const
    {
        std::string s;
        s += "plan_seq=" + std::to_string(req->plan_seq) + "\n";
        s += "node=" + std::to_string(n) + " pool=" +
            std::to_string(inv->pool_cardinality(n)) + " free=" +
            std::to_string(working.free_count(n)) + "\n";
        s += "need: task=" + std::to_string(need_.task) + " tid=" +
            std::to_string(need_.task_id) + " tile=" +
            std::to_string(need_.tile) + " chunk=" +
            std::to_string(need_.index) + " access=" +
            access_name(need_.access) + "\n";
        s += "blocker: " + need_.what + "\n";
        s += "tasks=" + std::to_string(req->tasks.size()) +
            " entry_bindings=" + std::to_string(entry.size()) +
            " ops_so_far=" + std::to_string(plan.queue.size()) + "\n";
        for (std::size_t q = 0;
             q + 1 < plan.queue.size() && q < 24; ++q)
        {
            PlanOp const &op = plan.queue[q];
            if (op.kind == PlanOpKind::Link)
            {
                s += "  L(" + std::to_string(op.u.link.tile) + "," +
                    std::to_string(op.u.link.index) + ")@n" +
                    std::to_string(op.u.link.node) + "c" +
                    std::to_string(op.u.link.chunk) + "\n";
            }
            else if (op.kind == PlanOpKind::Unlink)
            {
                s += "  U(" + std::to_string(op.u.unlink.tile) + "," +
                    std::to_string(op.u.unlink.index) + ")@n" +
                    std::to_string(op.u.unlink.node) + "c" +
                    std::to_string(op.u.unlink.chunk) + "\n";
            }
            else if (op.kind == PlanOpKind::Copy)
            {
                s += "  C(" + std::to_string(op.u.copy.tile) + "," +
                    std::to_string(op.u.copy.index) + ")n" +
                    std::to_string(op.u.copy.src.node) + "c" +
                    std::to_string(op.u.copy.src.chunk) + "->n" +
                    std::to_string(op.u.copy.dst.node) + "c" +
                    std::to_string(op.u.copy.dst.chunk) + "\n";
            }
            else
            {
                s += "  T()\n";
            }
        }
        // Every node's free count at the fault (the fold's occupancy
        // in one line — the census's own fold is the full render).
        s += "free: ";
        for (std::size_t m = 0; m < inv->node_count(); ++m)
        {
            s += std::to_string(m) + "=" +
                std::to_string(working.free_count(
                    static_cast<NodeId>(m))) +
                " ";
        }
        s += "\n";
        return s;
    }

    static char const *access_name(Access a)
    {
        switch (a)
        {
        case Access::Read:
            return "Read";
        case Access::Write:
            return "Write";
        case Access::ReadWrite:
            return "ReadWrite";
        }
        return "?";
    }

    // ---- chunk acquisition: the free list or the LRU victim ----

    // Growth asks succeed ONLY during the
    // beginning — before the pool's ONE growth episode has met its
    // wall (the soft limit, or the first allocation error, which the
    // engine freezes into the inventory's ceiling). Once the wall
    // stands, this returns false forever and the caller falls through
    // to the LRU recycling rule and, at
    // the working-set guard, the pool-exhaustion census — the budget
    // is fixed. When it does succeed, the working pool grows by
    // exactly the need and the final cardinality rides plan.growth
    // (the engine's commit runs the episode before the check).
    bool try_grow(NodeId n, std::size_t need)
    {
        std::size_t const card =
            working.pool_cardinality(n);
        std::size_t const limit = inv->pool_limit(n);
        if (limit != 0 && card + need > limit)
        {
            return false; // at (or past) the soft limit
        }
        working.grow_node(n, card + need);
        grow_slot_space(n, card + need);
        growth_final_[n] = card + need;
        return true;
    }


    // The slot space follows the pools: growing node n
    // shifts every later node's base, so the per-slot arrays re-base
    // exactly like DependencyState::grow_node — each old slot's live
    // state (recency link, stamp, pin, class bits) moves to its new
    // home, new slots start clean. The recency ORDERS carry chunk ids
    // local to their node, so rfront_/rback_ and the link chains need
    // no fixup.
    void grow_slot_space(NodeId n, std::size_t new_card)
    {
        std::size_t const nodes = inv->node_count();
        if (base_.size() < nodes + 1)
        {
            return; // arrays not built yet (pre-walk growth)
        }
        std::vector<std::size_t> const old_base = base_;
        std::vector<std::size_t> old_cards(nodes);
        for (std::size_t m = 0; m < nodes; ++m)
        {
            old_cards[m] = old_base[m + 1] - old_base[m];
        }
        if (new_card <= old_cards[n])
        {
            return;
        }
        // ONE pass: the grown node's new cardinality, every LATER
        // node's base shifts with it.
        for (std::size_t m = 0; m < nodes; ++m)
        {
            std::size_t const card =
                m == n ? new_card : old_cards[m];
            base_[m + 1] = base_[m] + card;
        }
        std::size_t const new_slots = base_.back();
        // Copy OLD extents only: the grown node's trailing slots start
        // fresh (the arrays above were value-initialized).
        auto rebase = [&](auto &vec, auto fresh) {
            std::remove_reference_t<decltype(vec)> nv(new_slots, fresh);
            for (std::size_t m = 0; m < nodes; ++m)
            {
                for (std::size_t c = 0; c < old_cards[m]; ++c)
                {
                    std::size_t const om = old_base[m] + c;
                    std::size_t const nm = base_[m] + c;
                    nv[nm] = vec[om];
                }
            }
            vec = std::move(nv);
        };
        rebase(rlink_, RLink{});
        rebase(stamp_, kSeedStamp);
        rebase(pinned_, static_cast<std::uint8_t>(0));
        rebase(pin_consumer_, std::size_t{0});
        rebase(tile_of_, kNoTileMark);
        rebase(deadcls_, static_cast<std::uint8_t>(0));
    }

    // A free chunk on node n, GROWING while the pool may grow, then
    // evicting LRU victims as needed. Each
    // eviction frees at least one chunk, so the loop ends unless the
    // census aborts it first. The just-evicted chunk is the next
    // pop_free (the free list is a derived view; push lands the back).
    // The miss path. `ask` is the growth act's size — the WHOLE
    // TILE at a tile's first miss (one segment per tile: its chunks
    // land VA-contiguous by construction), one chunk for the later
    // per-chunk misses.
    ChunkId acquire(NodeId n, std::size_t ask = 1)
    {
        for (;;)
        {
            std::optional<ChunkId> c = working.pop_free(n);
            if (c.has_value())
            {
                return *c;
            }
            // THE RULE, first: eviction on an exhausted
            // free list is LRU by last planned touch. Growth is the
            // LAST resort, after recycling cannot serve: only
            // when every linked chunk is the working set (no victim)
            // and no pin can serve is new memory truly REQUIRED —
            // then one TILE-SIZED ask, and only before the pool's
            // wall. This keeps growth in the beginning (the initial
            // working set) and the steady run allocating nothing:
            // grow-before-recycle would absorb every evicted replica
            // and pay a synchronizing allocation mid-loop, batch
            // after batch (a measurable per-run premium).
            std::optional<ChunkId> v = lru_victim(n);
            if (v.has_value())
            {
                evict_one(n, *v);
                continue;
            }
            if (spend_pinned(n))
            {
                continue;
            }
            if (try_grow(n, ask))
            {
                continue; // the grown chunks are on the free list
            }
            // Every linked chunk of n is the current task's working
            // set and the free list is empty: the need is uncoverable
            // (the front-of-order guard).
            need_.what =
                "the free list is empty and every linked chunk is "
                "the current task's working set";
            census_pool_exhaustion(n);
        }
    }

    // ---- the contiguous-run acquisition (the carried allocation
    //      discipline) ----

    // A multi-chunk tile's consumers are cheapest when its chain sits
    // at CONSECUTIVE chunk ids (the codelets' contiguity fast paths:
    // the gemm panel driver chains x-panels over k inside one cache
    // run, the copy descriptor and the elementwise bodies likewise);
    // a blind pop scatters the chain and every consumer pays the
    // staged gather. The walk resolves a tile's chunks index-ascending
    // in one pass, so the FIRST index reserves the whole run and the
    // rest consume it; the free list's pop discipline is the
    // planner's own (the blocks are interchangeable, deterministically
    // ordered), and this discipline picks consecutive ones
    // whenever the free list carries a full run. `reserved` holds the
    // first index's run base for the pass's later indexes (cleared at
    // the tile's last index; a base stolen by an interleaved landing
    // falls back to the blind pop for the rest of the tile).
    ChunkId acquire_for(NodeId n, ChunkIndex ix, ChunkIndex nchunks,
        std::optional<ChunkId> &reserved)
    {
        if (ix == 0)
        {
            // The contiguous-run handout for every tile (a run of one
            // included — it takes the lowest free chunk, a run EDGE,
            // so the single-chunk tiles never shred a longer run's
            // interior); a multi-chunk miss falls back to the blind
            // pop below.
            std::optional<ChunkId> run;
            {
                BrkTimer const brk(brk_on(), ns_pfr_);
                ++n_pfr_;
                run = working.pop_free_run(n, nchunks);
            }
            if (run.has_value())
            {
                reserved = nchunks > 1 ? *run : std::optional<ChunkId>{};
                return *run;
            }
            if (nchunks > 1)
            {
                // The fragmentation meter (diagnosis only): the
                // longest free run at a multi-chunk miss.
                std::size_t longest = working.longest_free_run(n);
                if (longest > run_miss_longest_)
                {
                    run_miss_longest_ = longest;
                }
                if (run_miss_want_ == 0 || nchunks > run_miss_want_)
                {
                    run_miss_want_ = nchunks;
                }
                ++run_misses_;
            }
            reserved = std::nullopt;
            return acquire(n, nchunks); // the tile's own segment
        }
        if (reserved.has_value())
        {
            ChunkId const want =
                static_cast<ChunkId>(*reserved + ix);
            // The continuation may name past the pool's
            // end: a seed binding (or a run reserved when the walk
            // held more) near the extent's edge has no base+ix chain
            // to complete — the guard falls back to the blind pop,
            // the run discipline being a preference, never a
            // requirement
            // (extents only grow, so inv's cardinality bounds the
            // working table's extent from below).
            if (want < *reserved + nchunks &&
                inv->chunk_valid(n, want) && working.free(n, want))
            {
                return want; // link() removes it from the free list
            }
            reserved = std::nullopt; // stolen mid-tile: fall back
        }
        return acquire(n);
    }

    // One eviction (pinned orders 4 and 5): the data-survival rule
    // first, then the UNLINK that frees the chunk. The survival rule,
    // complete: a victim that is the LAST actual replica of its
    // logical chunk is rescued unless its tile is DEAD — the in-batch
    // reader model alone is batch-scoped (a live tile whose readers
    // live in LATER batches would lose its bytes to a bare eviction);
    // the engine's dead set is the only tile whose bytes may die here.
    void evict_one(NodeId n, ChunkId v)
    {
        BrkTimer const brk(brk_on(), ns_evict_);
        evictions_ += 1;
        std::optional<std::pair<LogicalChunk, bool>> const held =
            working.at(n, v);
        LogicalChunk const lc = held->first;
        if (held->second && actual_replicas(lc, n).empty() &&
            (has_remaining_reader(lc.tile) || !tile_is_dead(lc.tile)))
        {
            rescue(n, v, lc); // the bytes MUST survive
        }
        emit_unlink(lc.tile, lc.index, n, v);
        // Order 7: the bytes now live somewhere else (rescued just now
        // or already replicated) — a batch reader ahead makes this a
        // pending fire (no reader, no host landing, or a write-only
        // future drops out at the enqueue or the drain's own checks).
        enqueue_pending(lc.tile, lc.index);
    }

    // The rescue landing (pinned order 5's target choice): the host
    // first (the drain direction), then devices ascending; within a
    // candidate, an existing OUTDATED binding of the logical chunk
    // (a COPY lands, no new binding) before a fresh LINK.
    void rescue(NodeId n, ChunkId v, LogicalChunk lc)
    {
        BrkTimer const brk(brk_on(), ns_rescue_);
        rescues_ += 1;
        need_.what = "evicting the last actual replica of tile " +
            std::to_string(lc.tile) + " chunk " +
            std::to_string(lc.index) + " needs a rescue landing";
        // Pass A: an existing outdated binding on a channel-bearing
        // node — the COPY's dst must be outdated; an
        // actual binding elsewhere contradicts the rescue's premise,
        // so every existing binding here is outdated.
        for (NodeId m : rescue_order(n))
        {
            if (!inv->has_channel(n, m))
            {
                continue;
            }
            std::optional<ChunkId> mb = working.binding_of(m, lc);
            if (mb.has_value())
            {
                emit_copy(lc.tile, lc.index, ChunkLoc{n, v},
                    ChunkLoc{m, *mb});
                return;
            }
        }
        // Pass B: a fresh LINK on a free chunk, then the COPY.
        // The fresh landing takes the candidate's LOWEST free
        // chunk (pop_free_run(m, 1)), never the free list's blind
        // LIFO back: consecutive rescues of one eviction burst then
        // land ASCENDING-CONSECUTIVE — the eviction COPY's dst extends
        // across full-class spans (the run merge), and the block
        // becomes the logical chunk's STABLE host home (Pass A reuses
        // it every later cycle), so the refetch side's src extends in
        // the same order too. The back pop handed out reverse-push-
        // order blocks — large d2h dst and h2d src gaps.
        // Address choice only (the blocks are interchangeable,
        // deterministically ordered); the victim order is untouched.
        for (NodeId m : rescue_order(n))
        {
            if (!inv->has_channel(n, m))
            {
                continue;
            }
            std::optional<ChunkId> mc;
            {
                BrkTimer const brk(brk_on(), ns_pfr_);
                ++n_pfr_;
                mc = working.pop_free_run(m, 1);
            }
            if (!mc.has_value())
            {
                // Growth is the LAST resort — only after
                // this candidate's recycling cannot serve (recycling
                // comes first): one chunk of new memory, and only
                // before the pool's wall.
                if (try_grow(m, 1))
                {
                    mc = working.pop_free_run(m, 1);
                }
            }
            if (!mc.has_value())
            {
                continue; // this candidate is full; the next may serve
            }
            emit_link(lc.tile, lc.index, m, *mc);
            emit_copy(lc.tile, lc.index, ChunkLoc{n, v},
                ChunkLoc{m, *mc});
            return;
        }
        // No landing anywhere: the rescue is uncoverable — the fault
        // is pool exhaustion (the plan-time fault), the census naming
        // the evicting node and the need.
        need_.what = "no node can land the rescue copy for the last "
                     "actual replica of tile " +
            std::to_string(lc.tile) + " chunk " +
            std::to_string(lc.index) + " (no channel-bearing node "
            "with an outdated binding or a free chunk)";
        census_pool_exhaustion(n);
    }

    // The rescue candidates: the host first, then devices ascending,
    // never the evicting node itself.
    mutable std::vector<NodeId> scratch_order_;
    std::vector<NodeId> const &rescue_order(NodeId n) const
    {
        std::vector<NodeId> &order = scratch_order_;
        order.clear();
        if (inv->host_id() != n)
        {
            order.push_back(inv->host_id());
        }
        for (std::size_t m = 0; m < inv->device_count(); ++m)
        {
            if (m != n)
            {
                order.push_back(static_cast<NodeId>(m));
            }
        }
        return order;
    }

    // ---- actual-data delivery (pinned order 3) ----

    // Bring ACTUAL data to node n's (possibly fresh) binding of lc.
    void fetch_actual(LogicalChunk lc, NodeId n, ChunkId dst)
    {
        BrkTimer const brk(brk_on(), ns_fetch_);
        refetches_ += 1;
        // Direct: the lowest NodeId with an actual binding and a
        // channel to n (devices ascending, the host last).
        for (std::size_t s = 0; s < inv->node_count(); ++s)
        {
            NodeId const src = static_cast<NodeId>(s);
            if (src == n)
            {
                continue;
            }
            std::optional<ChunkId> sb = working.binding_of(src, lc);
            if (sb.has_value() && working.flag_actual(src, *sb) &&
                inv->has_channel(src, n))
            {
                emit_copy(lc.tile, lc.index, ChunkLoc{src, *sb},
                    ChunkLoc{n, dst});
                return;
            }
        }
        // Staged through the host (the two-hop form). The source: the
        // lowest NodeId with an actual
        // binding — reachability is the host's business now.
        std::optional<ChunkLoc> src;
        for (std::size_t s = 0; s < inv->node_count(); ++s)
        {
            NodeId const m = static_cast<NodeId>(s);
            if (m == n)
            {
                continue;
            }
            std::optional<ChunkId> mb = working.binding_of(m, lc);
            if (mb.has_value() && working.flag_actual(m, *mb))
            {
                src = ChunkLoc{m, *mb};
                break;
            }
        }
        if (!src.has_value())
        {
            // No actual binding anywhere: the predicted world never
            // produced this data — a caller defect, not the plan-time
            // fault.
            throw std::runtime_error(
                "lru_planner: task " + std::to_string(need_.task) +
                " (tid " + std::to_string(need_.task_id) +
                ") reads tile " + std::to_string(lc.tile) + " chunk " +
                std::to_string(lc.index) +
                " whose data no actual binding holds anywhere — "
                "never produced in the predicted world");
        }
        NodeId const h = inv->host_id();
        std::optional<ChunkId> hb = working.binding_of(h, lc);
        if (hb.has_value() && working.flag_actual(h, *hb))
        {
            // Unreachable over a legal inventory (the host channels
            // every node, so an actual host binding is a direct
            // source); the belt keeps the walk honest anyway.
            emit_copy(lc.tile, lc.index, ChunkLoc{h, *hb},
                ChunkLoc{n, dst});
            return;
        }
        if (hb.has_value())
        {
            // The host's existing OUTDATED binding is the hop: land
            // the data on it, then forward — no LINK, no residue
            // UNLINK (the binding predates this walk).
            emit_copy(lc.tile, lc.index, *src, ChunkLoc{h, *hb});
            emit_copy(lc.tile, lc.index, ChunkLoc{h, *hb},
                ChunkLoc{n, dst});
            return;
        }
        // The fresh staging hop: LINK, two COPYs, the residue UNLINK.
        // The staging block takes the host's LOWEST free chunk, the same
        // ascending-run discipline as the rescue landing (the two-hop
        // COPYs merge across consecutive hops when the spans allow).
        need_.what = "the host staging hop for tile " +
            std::to_string(lc.tile) + " chunk " +
            std::to_string(lc.index) + " needs a free host chunk";
        std::optional<ChunkId> hc;
        {
            BrkTimer const brk(brk_on(), ns_pfr_);
            ++n_pfr_;
            hc = working.pop_free_run(h, 1);
        }
        if (!hc.has_value())
        {
            census_pool_exhaustion(h);
        }
        emit_link(lc.tile, lc.index, h, *hc);
        emit_copy(lc.tile, lc.index, *src, ChunkLoc{h, *hc});
        emit_copy(lc.tile, lc.index, ChunkLoc{h, *hc}, ChunkLoc{n, dst});
        emit_unlink(lc.tile, lc.index, h, *hc); // the staging residue
    }

    // ---- the operand resolution, one logical chunk ----

    // The tile-pass run reservation (acquire_for's slot): the operand
    // loop resolves one tile's chunks index-ascending in one pass, so
    // ONE slot serves the whole pass — set at index 0, consumed by
    // the later indexes, dropped at the tile's last index or the
    // moment a reserved chunk is stolen.
    std::optional<ChunkId> run_reserve_;

    Operand resolve(NodeId node, TileId tile, ChunkIndex index,
        Access access)
    {
        LogicalChunk const lc{tile, index};
        need_.tile = tile;
        need_.index = index;
        need_.access = access;
        std::optional<ChunkId> b = working.binding_of(node, lc);
        if (!b.has_value())
        {
            // No binding on the node: a concrete free chunk through
            // the state read — never "some free chunk" in the op
            //  — then LINK, which starts OUTDATED. The
            // chunk comes from the CONTIGUOUS-RUN discipline when the
            // tile carries several chunks and the free list holds a
            // full run (the consumers' fast paths ride on consecutive
            // ids; the pop discipline is the planner's own).
            ChunkIndex const n = tiles->n(tile);
            ChunkId const c =
                acquire_for(node, index, n, run_reserve_);
            if (index + 1 == n)
            {
                run_reserve_ = std::nullopt; // the pass's last index
            }
            emit_link(tile, index, node, c);
            b = c;
        }
        else if (index == 0)
        {
            // An existing binding (a replica read, or a chain whose
            // index 0 was FIRED early): no allocation here — but for a
            // multi-chunk tile the binding's CONTINUATION seeds the
            // pass's run reservation, so the still-missing indexes
            // prefer the chain's own slots (base+ix) when they sit
            // free and complete the fired run instead of scattering
            // anew (the staged-gemm cost of a shredded chain is the
            // price).
            ChunkIndex const n = tiles->n(tile);
            run_reserve_ = n > 1 ? *b : std::optional<ChunkId>{};
        }
        if (access != Access::Write &&
            !working.flag_actual(node, *b))
        {
            // Read and ReadWrite need ACTUAL data at the TILE_OP's
            // position; a Write never does — output
            // production overwrites the bytes.
            fetch_actual(lc, node, *b);
        }
        // The claim touch: the TILE_OP about to be emitted names this
        // chunk, so the task's planned touch starts here — a replica
        // read emits no earlier op, and without this touch a LATER
        // operand of the same task could evict the binding its own
        // TILE_OP is about to name (the working-set guard's claim
        // half; the emission's touch below is the same fact). The
        // claim also ends a pending fire's PIN: the prefetched binding
        // re-enters the recency order at this touch, evictable again
        // like any resident chunk (the fire's protection ran from the
        // fire to exactly here).
        touch(node, *b);
        clear_pin(node, *b);
        Operand op;
        op.tile = tile;
        op.index = index;
        op.access = access;
        op.node = node;
        op.chunk = *b;
        return op;
    }

    // ---- the dying-victim free (pinned order 6) ----

    // Free every binding of `tile` on every node, bare, at the tile's
    // dying position: the batch position after its last use (or the
    // queue head when this batch never uses it). No rescue ever — the
    // tile is dead, its bytes' only future is the drop, and no later
    // task of this batch names it (the schedule guarantees the
    // position). The bindings fold away exactly where the engine's
    // trailing retirement would have dropped them at the tail — the
    // free merely lands the chunks back on the free list IN time for
    // the walk's own needs instead of after them.
    void emit_dying_free(TileId tile)
    {
        ChunkIndex const n = tiles->n(tile);
        for (std::size_t m = 0; m < inv->node_count(); ++m)
        {
            NodeId const node = static_cast<NodeId>(m);
            for (ChunkIndex ix = 0; ix < n; ++ix)
            {
                std::optional<ChunkId> const b =
                    working.binding_of(node, LogicalChunk{tile, ix});
                if (b.has_value())
                {
                    emit_unlink(tile, ix, node, *b);
                    dying_frees_ += 1;
                }
            }
        }
    }

    // The dying schedule's emission point before batch position `i`:
    // every dead tile whose last use is position i-1 (or none, i == 0).
    void run_dying_frees(std::size_t i)
    {
        if (i >= dying_at_.size())
        {
            return;
        }
        for (TileId tile : dying_at_[i])
        {
            emit_dying_free(tile);
        }
    }

    // ---- the request validation (caller defects, loud) ----

    void validate_request()
    {
        // The entry: the plan's convention is ascending (node, chunk);
        // a duplicate chunk in the request is the caller's double-link.
        // The snapshot() the engine feeds R.seed already emits
        // ascending (node, chunk), so at every chained plan the sort
        // below is a no-op result — an is_sorted guard turns it into a
        // linear check (EXACT: std::sort over an already-sorted range
        // changes nothing; the guard only skips work whose output is
        // the input). A genuinely unsorted entry (a first plan, a
        // hand-built request) falls through to the sort as before.
        entry = req->entry;
        if (!std::is_sorted(entry.begin(), entry.end(),
                [](EntryBinding const &a, EntryBinding const &b) {
                    return a.node != b.node ? a.node < b.node
                                            : a.chunk < b.chunk;
                }))
        {
            std::sort(entry.begin(), entry.end(),
                [](EntryBinding const &a, EntryBinding const &b) {
                    return a.node != b.node ? a.node < b.node
                                            : a.chunk < b.chunk;
                });
        }
        for (std::size_t i = 1; i < entry.size(); ++i)
        {
            if (entry[i].node == entry[i - 1].node &&
                entry[i].chunk == entry[i - 1].chunk)
            {
                throw std::runtime_error(
                    "lru_planner: the request's entry names (node " +
                    std::to_string(entry[i].node) + ", chunk " +
                    std::to_string(entry[i].chunk) + ") twice");
            }
        }
        for (EntryBinding const &b : entry)
        {
            if (!tiles->carries(b.tile))
            {
                throw std::runtime_error(
                    "lru_planner: the entry state names tile " +
                    std::to_string(b.tile) +
                    ", which the tile table does not carry");
            }
            if (b.index >= tiles->n(b.tile))
            {
                throw std::runtime_error(
                    "lru_planner: the entry state names tile " +
                    std::to_string(b.tile) + " chunk " +
                    std::to_string(b.index) + ", but the tile carries " +
                    std::to_string(tiles->n(b.tile)) + " chunks");
            }
        }
        // A repeated TaskId in one batch is LEGAL (the
        // correspondence collapses each maximal RUN of consecutive
        // TILE_OPs of one task into one symbol, the symbol sequence
        // matching the task sequence one-for-one — a task the batch
        // submits twice appears as two symbols). This is the planner's
        // acceptance of the engine's MULTIPLIED TASK SYMBOLS:
        // one symbol = one (codelet, cl_args,
        // operand-shape-and-access) triple, many instances per batch
        // mapping to it — and because the ACCESS is part of the
        // engine's symbol key, one access per (symbol, tile) holds by
        // construction, the checker's per-(task, tile) access
        // presumption satisfied rather than violated.
        // Session scratch: the per-task tile
        // set and the born/retire books' per-task read/write sides are
        // cleared at each use — same membership semantics, no per-task
        // allocation (many tasks per steady plan).
        std::unordered_set<TileId> seen_tiles;
        for (PlannedTask const &t : req->tasks)
        {
            if (!inv->node_valid(t.node))
            {
                throw std::runtime_error(
                    "lru_planner: task " + std::to_string(t.task) +
                    " names node " + std::to_string(t.node) +
                    ", which the inventory does not carry");
            }
            seen_tiles.clear();
            for (TileUse const &u : t.operands)
            {
                if (!seen_tiles.insert(u.tile).second)
                {
                    // One entry per logical chunk is the operand list's
                    // own rule: a tile the task names
                    // twice must arrive as one ReadWrite use instead.
                    throw std::runtime_error(
                        "lru_planner: task " + std::to_string(t.task) +
                        " names tile " + std::to_string(u.tile) +
                        " twice — a read-and-written tile is one "
                        "ReadWrite use");
                }
                if (!tiles->carries(u.tile))
                {
                    throw std::runtime_error(
                        "lru_planner: task " + std::to_string(t.task) +
                        " names tile " + std::to_string(u.tile) +
                        ", which the tile table does not carry");
                }
            }
        }
    }

    // ---- the walk (the lowering, pinned orders 1 and 2) ----

    PlannedBatch run()
    {
        std::uint64_t const tpre0 = brk_on() ? brk_now() : 0;
        validate_request();

        // The working copy: seeded at the entry state, all-free pools
        // first (the inventory's cardinalities), the initial load
        // through the same transitions.
        std::vector<std::size_t> cards;
        cards.reserve(inv->node_count());
        for (std::size_t n = 0; n < inv->node_count(); ++n)
        {
            cards.push_back(inv->pool_cardinality(
                static_cast<NodeId>(n)));
        }
        working.init(cards);
        // The growth state's per-walk base: the live
        // cardinalities (the engine commits growth between plans, so
        // every walk starts at the real grown sizes).
        growth_base_ = cards;
        growth_final_ = cards;
        for (EntryBinding const &b : entry)
        {
            working.link(b.node, b.chunk, LogicalChunk{b.tile, b.index});
            if (b.actual)
            {
                working.actualize(b.node, b.chunk);
            }
        }

        // The liveness models: per tile, the last batch position that
        // READS it (Read or ReadWrite — a Write alone reads nothing)
        // for the data-survival rule, the last position that uses it
        // with ANY access for the dying free's position (a Write after
        // the last Read still needs the binding), and — order 7's key —
        // the FULL ascending schedule of reading positions for the
        // pending queue's next-use lookups. Built BEFORE the
        // recency machinery: the dead-class gate's dense rows (and
        // through them every recency push) read these models.
        // All four
        // models are DENSE rows over the request's tile ids (the hash
        // maps and hash-map-of-vectors were a large cost at giant
        // plans): one max-tile pass, one
        // count pass, one fill pass — the reader schedules land in a
        // CSR (uses_off_/uses_pos_), the last-read/last-use/dead rows
        // are arrays (kNoReadPos = absent). validate_request has
        // already thrown on a task naming one tile twice, so every
        // (task, tile) read-use operand contributes exactly one row
        // entry — the duplicate-use dedupe holds.
        {
            std::size_t max_tile = 0;
            for (EntryBinding const &b : entry)
            {
                max_tile = std::max(max_tile,
                    static_cast<std::size_t>(b.tile));
            }
            for (PlannedTask const &t : req->tasks)
            {
                for (TileUse const &u : t.operands)
                {
                    max_tile = std::max(max_tile,
                        static_cast<std::size_t>(u.tile));
                }
            }
            for (TileId tile : req->dead_tiles)
            {
                max_tile = std::max(max_tile,
                    static_cast<std::size_t>(tile));
            }
            dead_bit_.assign(max_tile + 1, 0);
            last_read_dense_.assign(max_tile + 1, kNoReadPos);
            last_use_dense_.assign(max_tile + 1, kNoReadPos);
            uses_off_.assign(max_tile + 2, 0);
            for (TileId tile : req->dead_tiles)
            {
                dead_bit_[static_cast<std::size_t>(tile)] = 1;
            }
            // Count the reader-schedule entries, then prefix.
            for (std::size_t i = 0; i < req->tasks.size(); ++i)
            {
                for (TileUse const &u : req->tasks[i].operands)
                {
                    if (u.access != Access::Write)
                    {
                        ++uses_off_[static_cast<std::size_t>(u.tile)
                            + 1];
                    }
                }
            }
            for (std::size_t t = 1; t < uses_off_.size(); ++t)
            {
                uses_off_[t] += uses_off_[t - 1];
            }
            uses_pos_.assign(uses_off_.back(), kNoReadPos);
            std::vector<std::size_t> fill(uses_off_.begin(),
                uses_off_.end() - 1); // per-tile cursor
            // Fill: positions ascend with i, so each row ascends by
            // construction; the last-read/last-use rows take the same
            // walk's maxima.
            for (std::size_t i = 0; i < req->tasks.size(); ++i)
            {
                for (TileUse const &u : req->tasks[i].operands)
                {
                    std::size_t const t =
                        static_cast<std::size_t>(u.tile);
                    if (u.access != Access::Write)
                    {
                        uses_pos_[fill[t]++] = i;
                        if (last_read_dense_[t] == kNoReadPos
                            || last_read_dense_[t] < i)
                        {
                            last_read_dense_[t] = i;
                        }
                    }
                    if (last_use_dense_[t] == kNoReadPos
                        || last_use_dense_[t] < i)
                    {
                        last_use_dense_[t] = i;
                    }
                }
            }
        }

        // The dying schedule (order 6): the dead set as an index, every
        // dead tile filed at its last use + 1 (position 0 when this
        // batch never uses it — the entry's dead residue).
        dying_at_.assign(req->tasks.size() + 1, {});
        for (TileId tile : req->dead_tiles)
        {
            std::size_t const use =
                last_use_dense_[static_cast<std::size_t>(tile)];
            dying_at_[use == kNoReadPos ? 0 : use + 1]
                .push_back(tile);
        }

        // The dead-class gate's flip schedule: every dead tile
        // this batch READS turns reader-passed at the boundary after
        // its last reading position — its still-linked bindings (if
        // any) join the counted class then. Filed per boundary, once
        // per tile; the task loop processes flip_at_[i] at the end of
        // task i, exactly when task_idx_ moves to i+1.
        flip_at_.assign(req->tasks.size() + 1, {});
        for (TileId tile : req->dead_tiles)
        {
            std::size_t const rd =
                last_read_dense_[static_cast<std::size_t>(tile)];
            if (rd != kNoReadPos)
            {
                flip_at_[rd].push_back(tile);
            }
        }

        // The recency machinery over the same slots.
        base_.assign(inv->node_count() + 1, 0);
        for (std::size_t n = 0; n < inv->node_count(); ++n)
        {
            base_[n + 1] = base_[n] + cards[n];
        }
        rlink_.assign(base_.back(), RLink{});
        stamp_.assign(base_.back(), kSeedStamp);
        rfront_.assign(inv->node_count(), kNoChunk);
        rback_.assign(inv->node_count(), kNoChunk);
        pinned_.assign(base_.back(), 0);
        pin_consumer_.assign(base_.back(), 0);
        // the dead-class gate's slot state (the flip schedule and
        // the dense mirrors are built above).
        tile_of_.assign(base_.back(), kNoTileMark);
        deadcls_.assign(base_.back(), 0);
        cls_count_.assign(inv->node_count(), 0);
        recency_seed();

        // The slack gate's books (order 7), once per plan: the fresh
        // write-only namings' demand per device node, and the dying
        // frees' guaranteed releases per device node. A dead tile this
        // batch never names releases at the plan head — before the
        // walk-start drain reads the free count — so it is in NO book
        // (a never-named dead tile is in no book). The
        // credit counts the tile's whole chunk count per device node:
        // whatever the walk produced is bounded by it, and every
        // mid-walk acquisition it covers was itself a counted demand —
        // the pairing these books rely on (a slight over-credit for a
        // multi-chunk tile partially resident on the node is possible
        // and bounded by n, the same bound a single-pool arithmetic
        // carries).
        born_prefix_.assign(inv->node_count(),
            std::vector<std::size_t>(req->tasks.size() + 2, 0));
        retire_prefix_.assign(inv->node_count(),
            std::vector<std::size_t>(req->tasks.size() + 2, 0));
        std::vector<TileId> reads_task;
        std::vector<TileId> writes_task;
        for (std::size_t i = 0; i < req->tasks.size(); ++i)
        {
            reads_task.clear();
            writes_task.clear();
            for (TileUse const &u : req->tasks[i].operands)
            {
                bool const reads = u.access != Access::Write;
                std::vector<TileId> &side =
                    reads ? reads_task : writes_task;
                if (std::find(side.begin(), side.end(), u.tile) ==
                    side.end())
                {
                    side.push_back(u.tile);
                }
            }
            for (TileId tile : writes_task)
            {
                if (std::find(reads_task.begin(), reads_task.end(),
                        tile) != reads_task.end())
                {
                    continue; // reads at this task: the pending side.
                }
                born_prefix_[req->tasks[i].node][i + 1] +=
                    tiles->n(tile);
            }
        }
        for (TileId tile : req->dead_tiles)
        {
            std::size_t const use =
                last_use_dense_[static_cast<std::size_t>(tile)];
            if (use == kNoReadPos)
            {
                continue; // the plan head already freed it.
            }
            for (std::size_t m = 0; m < inv->device_count(); ++m)
            {
                retire_prefix_[m][use + 1] += tiles->n(tile);
            }
        }
        for (std::size_t m = 0; m < inv->node_count(); ++m)
        {
            for (std::size_t k = 0; k + 1 < born_prefix_[m].size(); ++k)
            {
                born_prefix_[m][k + 1] += born_prefix_[m][k];
                retire_prefix_[m][k + 1] += retire_prefix_[m][k];
            }
        }

        // The dying free of the entry's dead residue rides the queue
        // head, before the first task's ops (position 0 of the walk).
        std::uint64_t const ns_pre = brk_on() ? brk_now() - tpre0 : 0;
        std::uint64_t const tstart0 = brk_on() ? brk_now() : 0;
        // The queue's one exact-size reservation before any
        // emission: a giant plan's doubling growth moved tens of MB
        // of PlanOp copies through the walk (a large cost at giant
        // plans). The bound is
        // loose and safe — every task emits at most its TILE_OP plus
        // per-chunk link/copy pairs, the entry and the dead set their
        // unlinks — and a reservation only changes capacity, never
        // content or order.
        plan.queue.reserve(req->tasks.size() * 4 + entry.size() * 2
            + req->dead_tiles.size() * 2 + 64);
        // The slack gate's trees: one per node over T[k] (the
        // MinTree block), built LAZILY at the node's first gate call —
        // small-plan workloads never call the gate, and their plans skip the
        // build entirely. gate_n_ = |tasks| + 1: k+1 indexes born's
        // last cell (both books carry |tasks| + 2 entries).
        gate_n_ = req->tasks.size() + 1;
        gate_trees_.resize(inv->node_count());
        gate_built_.assign(inv->node_count(), 0);
        run_dying_frees(0);

        // The walk-start queue (order 7): every logical chunk the
        // ENTRY holds ACTUAL somewhere but the device does not hold at
        // all — offloaded tiles returning, the
        // cross-batch ingress — is a pending fire for its next reading
        // consumer; then fire what the free capacity already fits,
        // before any kernel. (The dedup is a hash set keyed by the
        // packed logical chunk: the entry can carry thousands of
        // bindings at the no-pressure caps.)
        {
            std::unordered_set<std::uint64_t> actual_lcs;
            actual_lcs.reserve(entry.size() * 2 + 8);
            for (EntryBinding const &b : entry)
            {
                if (!b.actual)
                {
                    continue;
                }
                actual_lcs.insert(
                    (static_cast<std::uint64_t>(b.tile) << 32) |
                    static_cast<std::uint64_t>(b.index));
            }
            for (std::uint64_t key : actual_lcs)
            {
                enqueue_pending(static_cast<TileId>(key >> 32),
                    static_cast<ChunkIndex>(
                        key & 0xffffffffu));
            }
        }
        drain_pending();
        std::uint64_t const ns_start = brk_on() ? brk_now() - tstart0 : 0;

        // The lowering: task order, tile order given, indexes
        // ascending; ONE whole-tile TILE_OP per task.
        std::uint64_t const ttasks0 = brk_on() ? brk_now() : 0;
        for (std::size_t i = 0; i < req->tasks.size(); ++i)
        {
            PlannedTask const &t = req->tasks[i];
            task_idx_ = i;
            task_served_ = false;
            need_.task = i;
            need_.task_id = t.task;
            need_.what = "the TILE_OP operand needs a binding";
            std::vector<Operand> ops;
            std::size_t reserved = 0;
            for (TileUse const &u : t.operands)
            {
                ChunkIndex const n = tiles->n(u.tile);
                ops.reserve(reserved + n);
                reserved += n;
                for (ChunkIndex ix = 0; ix < n; ++ix)
                {
                    ops.push_back(resolve(t.node, u.tile, ix, u.access));
                }
            }
            emit_tile_op(t.task, t.node, std::move(ops));
            task_served_ = true;
            // The dying free of every dead tile whose last use was this
            // task (order 6): the chunks rejoin the free list here,
            // before the next task's needs.
            run_dying_frees(i + 1);
            // The dead-class gate's boundary flip: task_idx_
            // moves to i+1 now, so every dead tile whose last read was
            // task i turns reader-passed — its still-linked bindings
            // (if any) join the counted class. Probed through the LIVE
            // working table, so bindings the task just linked are seen
            // too; the dying frees above already removed theirs.
            if (i + 1 < flip_at_.size())
            {
                for (TileId tile : flip_at_[i])
                {
                    ChunkIndex const n = tiles->n(tile);
                    for (std::size_t m = 0; m < inv->node_count(); ++m)
                    {
                        NodeId const node = static_cast<NodeId>(m);
                        for (ChunkIndex ix = 0; ix < n; ++ix)
                        {
                            std::optional<ChunkId> const b =
                                working.binding_of(
                                    node, LogicalChunk{tile, ix});
                            if (!b.has_value())
                            {
                                continue;
                            }
                            std::size_t const s = slot(node, *b);
                            if (rlink_[s].in_list && deadcls_[s] == 0)
                            {
                                tile_of_[s] = tile;
                                deadcls_[s] = 1;
                                ++cls_count_[node];
                            }
                        }
                    }
                }
            }
            // The task boundary's free-list gain (order 7): fire what
            // the slack gate proves spare — the early refetches that
            // ride under the compute between here and their consumers.
            drain_pending();
        }

        std::uint64_t const ns_tasks = brk_on() ? brk_now() - ttasks0 : 0;
        std::uint64_t const texit0 = brk_on() ? brk_now() : 0;
        PlannedBatch out;
        out.plan.plan_seq = req->plan_seq;
        out.plan.entry = entry; // sorted by (node, chunk)
        // The growth this walk performed, per node, as
        // FINAL cardinalities (the executor commits the chunks before
        // the check; new chunks are free by construction). Nodes that
        // grew nothing stay at the walk's base and are omitted.
        for (std::size_t n = 0; n < growth_final_.size(); ++n)
        {
            if (growth_final_[n] != growth_base_[n])
            {
                if (out.plan.growth.size() < growth_final_.size())
                {
                    out.plan.growth.resize(growth_final_.size());
                }
                out.plan.growth[n] = growth_final_[n];
            }
        }
        out.plan.queue = std::move(plan.queue);
        out.exit = working.snapshot(); // the predicted exit fold
        out.stats.evictions = evictions_;
        out.stats.rescues = rescues_;
        out.stats.refetches = refetches_;
        out.stats.prefetches = prefetches_;
        out.stats.dying_frees = dying_frees_;
        out.stats.pin_spends = pin_spends_;
        out.stats.run_misses = run_misses_;
        out.stats.run_miss_want = run_miss_want_;
        out.stats.run_miss_longest = run_miss_longest_;
        out.stats.entry_bindings = entry.size();
        out.stats.exit_bindings = out.exit.size();
        if (brk_on())
        {
            std::uint64_t const ns_exit = brk_now() - texit0;
            std::fprintf(stderr,
                "[nnhaul-plan-brk] seq=%llu ops=%zu pre_ms=%.2f "
                "start_ms=%.2f tasks_ms=%.2f exit_ms=%.2f "
                "victim_ms=%.2f/v%llu vs%llu spend_ms=%.2f evict_ms=%.2f "
                "rescue_ms=%.2f enq_ms=%.2f drain_ms=%.2f "
                "gate_ms=%.2f/g%llu pfr_ms=%.2f/p%llu emit_ms=%.2f "
                "fetch_ms=%.2f\n",
                static_cast<unsigned long long>(req->plan_seq),
                out.plan.queue.size(),
                static_cast<double>(ns_pre) / 1.0e6,
                static_cast<double>(ns_start) / 1.0e6,
                static_cast<double>(ns_tasks) / 1.0e6,
                static_cast<double>(ns_exit) / 1.0e6,
                static_cast<double>(ns_victim_) / 1.0e6,
                static_cast<unsigned long long>(n_victim_),
                static_cast<unsigned long long>(n_vscan_),
                static_cast<double>(ns_spend_) / 1.0e6,
                static_cast<double>(ns_evict_) / 1.0e6,
                static_cast<double>(ns_rescue_) / 1.0e6,
                static_cast<double>(ns_enq_) / 1.0e6,
                static_cast<double>(ns_drain_) / 1.0e6,
                static_cast<double>(ns_gate_) / 1.0e6,
                static_cast<unsigned long long>(n_gate_),
                static_cast<double>(ns_pfr_) / 1.0e6,
                static_cast<unsigned long long>(n_pfr_),
                static_cast<double>(ns_emit_) / 1.0e6,
                static_cast<double>(ns_fetch_) / 1.0e6);
        }
        return out;
    }
};

} // namespace

// ---- the lowering walk's entry (see the header) ----

PlannedBatch LruPlanner::plan(PlanRequest const &request,
    NodeInventory const &inventory,
    TileTable const &tiles,
    bool validate)
{
    Walk w;
    w.inv = &inventory;
    w.tiles = &tiles;
    w.req = &request;
    PlannedBatch out = w.run();

    if (validate)
    {
        // The TEST convenience only (see the header): run the
        // one check on this walk's own product and demand the pass
        // plus the checker's exit fold equal the working copy's — the
        // same function run twice. The
        // executability table is stubbed permissively from the
        // request: each task runnable exactly on its named node (the
        // registry fact is the registration piece's, outside
        // the planner's inputs). The chain input is absent: the
        // request's entry state stands on its own rows (a cold start
        // as far as the walk knows; the ENGINE supplies the chain at
        // its own check).
        std::vector<SeqTask> seq = to_seq_tasks(request.tasks);
        ExecutabilityTable runnable;
        std::unordered_set<TaskId> added;
        for (PlannedTask const &t : request.tasks)
        {
            // A repeated symbol registers its row once (many instances
            // may map to one symbol).
            if (added.insert(t.task).second)
            {
                runnable.add(t.task, std::vector<NodeId>{t.node});
            }
        }
        // The walk may have GROWN pools beyond the live
        // inventory (the engine commits the growth at issue time);
        // the debug check validates against the walk's FINAL
        // cardinalities, so clone the inventory grown to them.
        NodeInventory grown = inventory;
        for (std::size_t n = 0; n < w.growth_final_.size(); ++n)
        {
            if (w.growth_final_[n] != inventory.pool_cardinality(
                                         static_cast<NodeId>(n)))
            {
                grown.grow_node(
                    static_cast<NodeId>(n), w.growth_final_[n]);
            }
        }
        CheckInput input;
        input.plan = &out.plan;
        input.inventory = &grown;
        input.tiles = &tiles;
        input.runnable = &runnable;
        input.task_sequence = &seq;
        CheckResult result = check(input);
        if (!result.pass)
        {
            throw std::runtime_error(
                "lru_planner: the walk's own product failed its "
                "debug check: " +
                result.fault);
        }
        if (result.exit != out.exit)
        {
            throw std::runtime_error(
                "lru_planner: the walk's working copy and the "
                "checker's fold disagree on the exit state");
        }
    }
    return out;
}

} // namespace nnhaul
