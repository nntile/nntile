/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file src/host_pool.hh
 * The host node's pool only: whole chunk_class_bytes() blocks pinned on
 * demand, addr() over the block table, and the census counts the loud
 * failures and test_hooks read. Residency is NOT here — it lives in
 * the binding table; the pool owns bytes, nothing else.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

// The host side of the node inventory: the host node's pool IS pinned
// pages — the host's implementation of the backend contract, with
// DMA-stable addresses.
//
// NO preallocated memory by default: the pool starts with ONLY the
// blocks an explicit preallocation ask names (NNHAUL_PREALLOC_CPU_MEM;
// default zero, CHECKED against the limit at init) and grows at
// planning time through the TWO-PHASE ladder — every allocation event
// TILE-WISE first (one cudaHostAlloc of the whole demanding tile's
// blocks, pinned and faulted by its own allocation), single blocks for
// that event only on fallback, a failed single-block allocation
// flipping the pool NON-ALLOCABLE permanently — until the SOFT limit
// (limit_blocks: the NNHAUL_LIMIT_CPU_MEM knob, else a positive
// cpu_cap_bytes init argument, else 0 = unlimited) or the flip; the
// run then works within the budget it has, and nothing is freed until
// the pool's death. A block's address is VALID and STABLE from its
// allocation to the pool's death (blocks never move); free/used
// accounting is the binding table's business, not this pool's.
// Single-threaded by contract — the engine's locks guard it; the pool
// carries none.

#include <nnhaul/defs.h>

#include <cstddef>
#include <string>
#include <vector>

namespace nnhaul
{

// ---- the garbage-knob fallback ----

// The fallback LIMIT, in MiB: 4096 (kHostPoolDefaultMiB). It fires
// only for a SET limit knob that is not a whole number of MiB, after
// its one warning. 4096 MiB is 1024 blocks of the 4 MiB class by
// default.
inline constexpr std::size_t kHostPoolDefaultMiB = 4096;

// ---- the census counts  ----

// What the pool hands test_hooks and the loud-failure census: the
// grown-pool facts — blocks held, the soft limit, the bytes. The FREE
// count the hooks report comes from the binding table, never from
// here. Data only — the emitter prints it and aborts; a failure is
// never serviced.
struct HostPoolStats
{
    std::size_t blocks = 0;                     // blocks held (grown so far).
    std::size_t limit_blocks = 0;               // the soft ceiling (0 = unlimited).
    std::size_t block_bytes = chunk_class_bytes(); // the one class.
    std::size_t pool_bytes = 0;                 // blocks * block_bytes.
    std::size_t hard_capped = 0;                // 1 once allocations closed.

    // One loud line for the census: every count, greppable.
    std::string str() const;
};

// ---- the host node's pool ----

class HostPool
{
  public:
    // `limit_blocks` is the SOFT ceiling (0 = unlimited: growth ends
    // only at the non-allocable flip); `initial_blocks` is the
    // explicit preallocation head start (default 0 — preallocate
    // nothing unless asked; NNHAUL_PREALLOC_CPU_MEM names it), CHECKED
    // against the limit (a larger ask is a loud init
    // fault, never clamped). Throws std::runtime_error, census-style,
    // when the PREALLOCATION itself fails or under-delivers (an ask
    // the machine cannot fully honor at init is a configuration
    // fault, loud); ordinary growth later stops at the soft wall or
    // the non-allocable flip, never thrown.
    HostPool(std::size_t limit_blocks, std::size_t initial_blocks = 0);

    // Every block leaves: cudaFreeHost per block (a diagnostic stderr
    // line if a free fails — a destructor never throws).
    ~HostPool();

    HostPool(HostPool const &) = delete;
    HostPool &operator=(HostPool const &) = delete;

    // One block's address — VALID and STABLE from the block's
    // allocation to the pool's death (blocks never move). Throws
    // std::runtime_error on an index the pool does not hold (loud,
    // for a defect).
    void *addr(std::size_t block) const;

    // The TWO-PHASE allocation rule. In the
    // ALLOCABLE phase every allocation event tries TILE-WISE first
    // (one segment for the whole demanding tile) and falls back to
    // SINGLE blocks for that event only when tile-wise fails; success
    // by either path stays allocable (the next event tries tile-wise
    // again). A failed SINGLE-block allocation — not even one chunk
    // fits — flips the pool to NON-ALLOCABLE, permanently. The soft
    // limit bounds every event (grow is possible up to the limit);
    // nothing is freed until the pool's death.
    std::size_t alloc_event(std::size_t chunks);

    // The planning-time growth: the demanding tile through the ladder,
    // then (under a soft limit) the remaining budget as one more event;
    // an unlimited pool serves the demand only. Idempotent at the
    // walls. Returns how many blocks were added.
    std::size_t grow_to_wall(std::size_t tile_chunks);

    // ENSURE wrapper for the init-time preallocation ask: at least
    // `n` TOTAL blocks, through alloc_event's ladder.
    std::size_t grow(std::size_t n);

    bool growth_closed() const // NON-ALLOCABLE: allocations are off
    {
        return hard_capped_;
    }

    std::size_t block_count() const;  // blocks held (grown so far).
    std::size_t limit_blocks() const; // the soft ceiling (0 = unlimited).
    std::size_t segment_count() const // driver allocations (one per grow).
    {
        return segments_.size();
    }
    std::size_t block_bytes() const;  // == chunk_class_bytes(), the one class.
    std::size_t pool_bytes() const;   // block_count() * block_bytes().
    bool hard_capped() const;         // the hard limit fired.

    // The census counts: the grown-pool facts.
    HostPoolStats stats() const;

    // The SOFT-limit precedence (first match wins, whole class blocks
    // only, 0 = unlimited):
    //   (1) NNHAUL_LIMIT_CPU_MEM set -> that many MiB;
    //   (2) else cpu_cap_bytes > 0 -> that many bytes;
    //   (3) else 0 — unlimited: growth ends at the hard limit only.
    // A knob that is set but not a whole number of MiB keeps the old
    // chain's discipline: ONE stderr warning per process, then the
    // kHostPoolDefaultMiB fallback as the LIMIT. Read per call, so
    // scenarios can set and unset the knob around inits. Pure —
    // touches no device.
    static std::size_t resolve_limit_blocks(std::size_t cpu_cap_bytes);

    // The contiguous span from block `block` to its
    // SEGMENT's end, in BYTES: an acquire (one pointer for a whole
    // tile) and a merged driver copy must stay inside one allocation —
    // per-allocation VA contiguity is the contract segments provide.
    std::size_t contig_bytes(std::size_t block) const;

  protected:
    // One allocation attempt of exactly `seg_blocks` blocks; false
    // WITHOUT the mode decision (the ladder owns it). Test seam:
    // overridable so the suite can script allocation failures (the
    // ladder's fallback and flip paths are otherwise unreachable).
    virtual bool allocate_segment(std::size_t seg_blocks);

  private:

    // The mode flip: a single-block allocation failed — the pool is
    // non-allocable, permanently. One diagnostic line per process.
    void enter_non_allocable();

    // Segments: one cudaHostAlloc of WHOLE BLOCKS each —
    // TILE-SIZED when the ladder's first rung lands (a tile's blocks
    // contiguous in one allocation), single-block in the fallback —
    // appended on demand, nothing at init. A segment's blocks are
    // VA-contiguous; blocks_ flattens them for O(1) addr().
    struct Segment
    {
        void *base = nullptr;
        std::size_t blocks = 0;
    };
    std::vector<Segment> segments_;
    std::vector<void *> blocks_;
    std::size_t limit_blocks_ = 0;
    bool hard_capped_ = false;

  public:
};

} // namespace nnhaul
