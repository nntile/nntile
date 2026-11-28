/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file src/host_pool.cc
 * The host node's pool's acts: the soft-limit precedence resolution,
 * per-block pinned allocation with the pinning verification, the
 * growth ladder, addr() over the block table, the census counts, and
 * the per-block cudaFreeHost teardown declared in host_pool.hh.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include "host_pool.hh"

#include <cstring>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace nnhaul
{

namespace
{

// One strict env parse (the env-knob discipline, inlined so this
// pool pulls in no other header): a whole decimal integer, no
// trailing garbage, no out-of-range (strtol clamps to LONG_MAX and
// sets ERANGE).
bool parse_mib(char const *e, long &out)
{
    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(e, &end, 10);
    if (end == e || *end != '\0' || errno == ERANGE)
    {
        return false;
    }
    out = parsed;
    return true;
}

// The garbage-knob fallback in blocks: kHostPoolDefaultMiB of
// whole class blocks, floored at one (a class larger than the fallback
// still backs one block).
std::size_t fallback_blocks()
{
    return std::max<std::size_t>(
        1, (kHostPoolDefaultMiB << 20) / chunk_class_bytes());
}

} // namespace

// ---- the census counts ----

std::string HostPoolStats::str() const
{
    return "host pool census: blocks=" + std::to_string(blocks) +
           " limit_blocks=" + (limit_blocks == 0
                                   ? std::string("unlimited")
                                   : std::to_string(limit_blocks)) +
           " block_bytes=" + std::to_string(block_bytes) +
           " pool_bytes=" + std::to_string(pool_bytes) +
           " hard_capped=" + std::to_string(hard_capped);
}

// ---- the soft-limit precedence ----

std::size_t HostPool::resolve_limit_blocks(std::size_t cpu_cap_bytes)
{
    // (1) The NNHAUL_LIMIT_CPU_MEM knob (MiB) is the SOFT CEILING of
    // the host (CPU-RAM) pool whenever set, floored to one class block
    // — a 17 MiB limit allows four 4 MiB blocks, a sub-class limit
    // one.
    // (No old-name alias: old knob names are deleted, not carried.)
    char const *knob = std::getenv("NNHAUL_LIMIT_CPU_MEM");
    if (knob != nullptr && *knob != '\0')
    {
        long mib = 0;
        if (parse_mib(knob, mib) && mib >= 0 &&
            static_cast<unsigned long>(mib) <=
                static_cast<unsigned long>(
                    std::numeric_limits<std::size_t>::max() >> 20))
        {
            return std::max<std::size_t>(
                1, (static_cast<std::size_t>(mib) << 20) / chunk_class_bytes());
        }

        // Garbage: ONE warning per process, then the
        // kHostPoolDefaultMiB fallback as the LIMIT (strict parse, no
        // silent clamping).
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::fprintf(stderr,
                "[nnhaul] NNHAUL_LIMIT_CPU_MEM=%s is not a whole number "
                "of MiB; falling back to %zu MiB\n",
                knob,
                kHostPoolDefaultMiB);
        }
        return fallback_blocks();
    }

    // (2) The init argument: cpu_cap_bytes is the API
    // ceiling — floored likewise, whole blocks only.
    if (cpu_cap_bytes > 0)
    {
        return std::max<std::size_t>(1, cpu_cap_bytes / chunk_class_bytes());
    }

    // (3) Unlimited: growth ends at the HARD limit
    // (cudaHostAlloc failure) only. The old default
    // (max(512, 2 x arena chunks)) is deleted machinery — no
    // preallocated or pre-sized memory without an explicit ask.
    return 0;
}

// ---- the pool's acts ----

HostPool::HostPool(std::size_t limit_blocks, std::size_t initial_blocks)
    : limit_blocks_(limit_blocks)
{
    // Preallocation happens AT INIT and is
    // CHECKED against the soft limit: a head start greater than the
    // limit is a configuration fault, loud, never serviced (an ask
    // the machine cannot honor at init is loud the same way).
    if (limit_blocks_ != 0 && initial_blocks > limit_blocks_)
    {
        throw std::runtime_error(
            "host pool: the explicit preallocation of " +
            std::to_string(initial_blocks) +
            " blocks exceeds the soft limit of " +
            std::to_string(limit_blocks_) +
            " blocks — raise the limit or lower the preallocation");
    }
    std::size_t const ask = initial_blocks;
    if (ask > 0 && grow(ask) < ask) // partial delivery is a fault too

    {
        HostPoolStats const stats = this->stats();
        throw std::runtime_error(
            std::string("host pool: the explicit preallocation of ") +
            std::to_string(ask) + " blocks failed (" + stats.str() + ")");
    }
}

// The allocation ladder for ONE event of
// `chunks` blocks: TILE-WISE first — one segment for the whole ask;
// on error, SINGLE blocks for THIS event only. Success by either
// path keeps the pool in the ALLOCABLE phase (the next event tries
// tile-wise again); a failed SINGLE-block allocation — not even one
// chunk fits — flips the pool to NON-ALLOCABLE, permanently.
std::size_t HostPool::alloc_event(std::size_t chunks)
{
    if (chunks == 0 || hard_capped_)
    {
        return 0;
    }
    std::size_t remaining = limit_blocks_ != 0
        ? limit_blocks_ - blocks_.size()
        : std::numeric_limits<std::size_t>::max();
    if (remaining == 0)
    {
        return 0; // the soft wall: the budget is fixed
    }
    std::size_t const want = std::min(chunks, remaining);
    if (allocate_segment(want))
    {
        return want; // tile-wise landed whole
    }
    if (want <= 1)
    {
        enter_non_allocable();
        return 0;
    }
    std::size_t added = 0;
    while (added < want)
    {
        remaining = limit_blocks_ != 0
            ? limit_blocks_ - blocks_.size()
            : std::numeric_limits<std::size_t>::max();
        if (remaining == 0)
        {
            break; // the soft wall mid-event: the budget is fixed
        }
        if (!allocate_segment(1))
        {
            enter_non_allocable(); // not even one block: allocations off
            break;
        }
        added += 1;
    }
    return added;
}

std::size_t HostPool::grow_to_wall(std::size_t tile_chunks)
{
    // The planning-time growth: the demanding
    // tile through the ladder; a pool under a soft limit then takes
    // the REMAINING budget as one more ladder event (grow is possible
    // up to the limit, all during planning); an unlimited pool serves
    // the demand only — no wall to seek, a single-block failure will
    // flip it non-allocable. Nothing is freed until the pool's death.
    if (tile_chunks == 0 || hard_capped_)
    {
        return 0;
    }
    std::size_t const before = blocks_.size();
    (void) alloc_event(tile_chunks);
    if (limit_blocks_ != 0 && !hard_capped_
        && blocks_.size() < limit_blocks_)
    {
        (void) alloc_event(limit_blocks_ - blocks_.size());
    }
    return blocks_.size() - before;
}

std::size_t HostPool::grow(std::size_t n)
{
    // ENSURE-exactly-n for the init-time preallocation ask: the ladder
    // (tile-wise first, single blocks on error), non-allocable on a
    // single-block failure.
    if (blocks_.size() >= n)
    {
        return 0;
    }
    return alloc_event(n - blocks_.size());
}

// One allocation attempt of exactly `seg_blocks` whole blocks (the
// pin verification and the fault-in ride along). Returns false
// WITHOUT the mode decision (the ladder owns it).
bool HostPool::allocate_segment(std::size_t seg_blocks)
{
    if (seg_blocks == 0)
    {
        return false;
    }
    std::size_t const bytes = seg_blocks * chunk_class_bytes();
    void *seg = nullptr;
    if (cudaHostAlloc(&seg, bytes, cudaHostAllocDefault) != cudaSuccess
        || seg == nullptr)
    {
        return false;
    }
    // The pinning VERIFICATION (DMA-stability), both ends.
    unsigned flags = 0;
    if (cudaHostGetFlags(&flags, seg) != cudaSuccess
        || (seg_blocks > 1
            && cudaHostGetFlags(&flags,
                   static_cast<char *>(seg) + bytes - chunk_class_bytes())
                   != cudaSuccess))
    {
        cudaFreeHost(seg);
        return false;
    }
    // Fault the fresh segment by its own allocation: allocated ==
    // resident == pinned.
    std::memset(seg, 0, bytes);
    segments_.push_back(Segment{seg, seg_blocks});
    blocks_.reserve(blocks_.size() + seg_blocks);
    for (std::size_t b = 0; b < seg_blocks; ++b)
    {
        blocks_.push_back(static_cast<char *>(seg) + b * chunk_class_bytes());
    }
    return true;
}

void HostPool::enter_non_allocable()
{
    if (!hard_capped_)
    {
        hard_capped_ = true;
        std::fprintf(stderr,
            "[nnhaul] host pool: a single-block allocation failed — "
            "not even one chunk fits; the pool is NON-ALLOCABLE from "
            "%zu blocks on, permanently\n",
            blocks_.size());
    }
}

HostPool::~HostPool()
{
    // Segments free as segments: one cudaFreeHost per
    // allocation (a block is an interior pointer, never freed alone).
    for (Segment &seg : segments_)
    {
        if (cudaFreeHost(seg.base) != cudaSuccess)
        {
            std::fprintf(stderr,
                "[nnhaul] host pool: cudaFreeHost of a %zu-block "
                "segment failed (diagnostic; teardown continues)\n",
                seg.blocks);
        }
    }
}

void *HostPool::addr(std::size_t block) const
{
    if (block >= blocks_.size())
    {
        throw std::runtime_error(
            "host pool: block " + std::to_string(block) +
            " is not held (the pool grew " +
            std::to_string(blocks_.size()) + " blocks) — a defect");
    }
    return blocks_[block];
}

std::size_t HostPool::contig_bytes(std::size_t block) const
{
    // Walk the segments: the span from this block to its segment's
    // end (the one allocation it lives in).
    std::size_t base = 0;
    for (Segment const &seg : segments_)
    {
        if (block < base + seg.blocks)
        {
            return (base + seg.blocks - block) * chunk_class_bytes();
        }
        base += seg.blocks;
    }
    return 0; // not held
}

std::size_t HostPool::block_count() const
{
    return blocks_.size();
}

std::size_t HostPool::limit_blocks() const
{
    return limit_blocks_;
}

std::size_t HostPool::block_bytes() const
{
    return chunk_class_bytes();
}

std::size_t HostPool::pool_bytes() const
{
    return blocks_.size() * chunk_class_bytes();
}

bool HostPool::hard_capped() const
{
    return hard_capped_;
}

HostPoolStats HostPool::stats() const
{
    HostPoolStats s;
    s.blocks = blocks_.size();
    s.limit_blocks = limit_blocks_;
    s.block_bytes = chunk_class_bytes();
    s.pool_bytes = blocks_.size() * chunk_class_bytes();
    s.hard_capped = hard_capped_ ? 1 : 0;
    return s;
}

} // namespace nnhaul
