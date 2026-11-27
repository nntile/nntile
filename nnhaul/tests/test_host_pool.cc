/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine: a virtual planner
 * builds one plan per batch from a drained snapshot, and a literal
 * executor issues exactly the planned operations.
 *
 * @file tests/test_host_pool.cc
 * The host node's pinned pool: the precedence
 * (the knob floored to one class block, garbage warning once
 * into the 4096-MiB fallback, the cpu_cap_bytes cap floored
 * likewise, the carried-over max(512, 2 x arena chunks) default,
 * uncapped), the zero-block loud error, and — with a
 * CUDA context — the one pinned carve: construction yields the
 * requested whole blocks, addr() stable and distinct per block, the
 * census counts, and the loud foreign index.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#include "../src/host_pool.hh"
#include "check.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

using namespace nnhaul;

void set_env(char const *name, char const *value) { setenv(name, value, 1); }

void unset_env(char const *name) { unsetenv(name); }

// Capture stderr into a temp file while the guard lives (dup2), read
// it back on demand — the warns-once check must SEE the one warning.
class StderrCapture
{
  public:
    StderrCapture() : saved_(dup(STDERR_FILENO)), tmp_(std::tmpfile())
    {
        std::fflush(stderr);
        dup2(fileno(tmp_), STDERR_FILENO);
    }

    std::string read_back()
    {
        std::fflush(stderr);
        dup2(saved_, STDERR_FILENO);
        ::close(saved_);
        saved_ = -1;
        std::string out;
        std::fseek(tmp_, 0, SEEK_SET);
        char buf[512];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof buf, tmp_)) > 0)
        {
            out.append(buf, n);
        }
        std::fclose(tmp_);
        tmp_ = nullptr;
        return out;
    }

    ~StderrCapture()
    {
        if (saved_ >= 0)
        {
            dup2(saved_, STDERR_FILENO);
            ::close(saved_);
        }
        if (tmp_ != nullptr)
        {
            std::fclose(tmp_);
        }
    }

    StderrCapture(StderrCapture const &) = delete;
    StderrCapture &operator=(StderrCapture const &) = delete;

  private:
    int saved_ = -1;
    std::FILE *tmp_ = nullptr;
};

std::size_t count_occurrences(std::string const &hay, char const *needle)
{
    std::size_t n = 0;
    for (std::size_t at = hay.find(needle); at != std::string::npos;
         at = hay.find(needle, at + 1))
    {
        n += 1;
    }
    return n;
}

// Precedence 1's garbage-knob fallback in blocks, computed the pool's way
// so the expectations below hold for every configured chunk class.
std::size_t fallback_blocks()
{
    return std::max<std::size_t>(
        1, (kHostPoolDefaultMiB << 20) / chunk_class_bytes());
}

} // namespace

int main()
{
    // ---- precedence 1: the knob, floored to one class block ----

    set_env("NNHAUL_LIMIT_CPU_MEM", "64");
    CHECK(HostPool::resolve_limit_blocks(0) ==
          (std::size_t{64} << 20) / chunk_class_bytes());
    // The knob wins over both other inputs (first match wins).
    CHECK(HostPool::resolve_limit_blocks(std::size_t{1} << 30) ==
          (std::size_t{64} << 20) / chunk_class_bytes());
    // Whole blocks only: 17 MiB buys four 4 MiB blocks.
    set_env("NNHAUL_LIMIT_CPU_MEM", "17");
    CHECK(HostPool::resolve_limit_blocks(0) ==
          (std::size_t{17} << 20) / chunk_class_bytes());
    // The floor at one class block: a sub-class knob still backs one
    // copy .
    set_env("NNHAUL_LIMIT_CPU_MEM", "1");
    CHECK(HostPool::resolve_limit_blocks(0) == 1);
    set_env("NNHAUL_LIMIT_CPU_MEM", "0");
    CHECK(HostPool::resolve_limit_blocks(0) == 1);

    {
        // Garbage: the fallback's blocks, and exactly ONE
        // warning for two resolutions (the once-flag is per process,
        // so this must be the FIRST garbage scenario).
        StderrCapture cap;
        set_env("NNHAUL_LIMIT_CPU_MEM", "notanumber");
        std::size_t const a = HostPool::resolve_limit_blocks(0);
        std::size_t const b = HostPool::resolve_limit_blocks(0);
        std::string const err = cap.read_back();
        CHECK(a == fallback_blocks());
        CHECK(b == fallback_blocks());
        CHECK(count_occurrences(err, "[nnhaul] NNHAUL_LIMIT_CPU_MEM=") == 1);
    }

    // More garbage keeps the fallback silently (the once-flag is
    // already set): trailing characters, negatives, out-of-range.
    set_env("NNHAUL_LIMIT_CPU_MEM", "16MiB");
    CHECK(HostPool::resolve_limit_blocks(0) == fallback_blocks());
    set_env("NNHAUL_LIMIT_CPU_MEM", "-8");
    CHECK(HostPool::resolve_limit_blocks(0) == fallback_blocks());
    set_env("NNHAUL_LIMIT_CPU_MEM", "99999999999999999999");
    CHECK(HostPool::resolve_limit_blocks(0) == fallback_blocks());

    // ---- precedence 2: cpu_cap_bytes, floored likewise ----

    unset_env("NNHAUL_LIMIT_CPU_MEM");
    std::size_t const cap128 = std::size_t{128} << 20;
    CHECK(HostPool::resolve_limit_blocks(cap128) == cap128 / chunk_class_bytes());
    std::size_t const cap17 = std::size_t{17} << 20;
    CHECK(HostPool::resolve_limit_blocks(cap17) ==
          cap17 / chunk_class_bytes());
    // The floor at one class block, byte-exact.
    CHECK(HostPool::resolve_limit_blocks(1) == 1);

    // ---- the unlimited case: no limit (the memory manager default) ----

    // A zero cap and no knob: NO soft limit — growth ends at the hard
    // limit only (the old max(512, 2 x arena) pre-carved default is
    // deleted machinery; no preallocated or pre-sized memory without
    // an explicit ask).
    CHECK(HostPool::resolve_limit_blocks(0) == 0);

    // ---- the preallocation-above-limit loud error (policy: an
    // init-time head start is CHECKED, never clamped) ----

    {
        bool threw = false;
        try
        {
            HostPool pool(4, 8); // an 8-block head start over a
                                 // 4-block soft limit: config fault
        }
        catch (std::runtime_error const &e)
        {
            threw = true;
            CHECK(std::string(e.what()).find("exceeds the soft limit") !=
                  std::string::npos);
        }
        CHECK(threw);
    }

    // ---- the empty pool is lawful (no CUDA touched) ----

    {
        // A pool that starts EMPTY: block_count 0, no
        // allocation, addr() loud on any index — growth supplies the
        // blocks later (the old zero-block refusal is deleted
        // machinery: preallocate nothing unless asked).
        HostPool pool(0);
        CHECK(pool.block_count() == 0);
        CHECK(pool.pool_bytes() == 0);
        CHECK(!pool.hard_capped());
        bool threw = false;
        try
        {
            (void) pool.addr(0);
        }
        catch (std::runtime_error const &e)
        {
            threw = true;
        }
        CHECK(threw);
    }

    // ---- the pinned carve, on a real CUDA context ----

    // The carve needs CUDA (cudaHostAlloc): skip loudly with a
    // printed notice and exit 0 when no context exists — the resolution
    // steps above have already run.
    if (cudaFree(0) != cudaSuccess)
    {
        std::cout << "skip: no CUDA context (the cudaFree(0) probe "
                     "failed) -- the pinned-carve tests need one\n";
        unset_env("NNHAUL_LIMIT_CPU_MEM");
        return g_checks_failed == 0 ? 0 : 1;
    }

    {
        // Construction yields the requested whole blocks, the census
        // counts agree, addr() is stable and distinct per block with
        // class spacing, the bytes are writable pinned host memory,
        // and a foreign index is loud.
        set_env("NNHAUL_LIMIT_CPU_MEM", "64");
        std::size_t const want = (std::size_t{64} << 20) / chunk_class_bytes();
        // The preallocated head start: the limit ceilings growth, the
        // initial ask supplies the blocks at construction.
        HostPool pool(HostPool::resolve_limit_blocks(0), want);
        CHECK(pool.block_count() == want);
        CHECK(pool.block_bytes() == chunk_class_bytes());
        CHECK(pool.pool_bytes() == want * chunk_class_bytes());
        HostPoolStats const s = pool.stats();
        CHECK(s.blocks == want);
        CHECK(s.block_bytes == chunk_class_bytes());
        CHECK(s.pool_bytes == want * chunk_class_bytes());
        CHECK(s.str().find("blocks=" + std::to_string(want)) !=
              std::string::npos);
        std::fprintf(stderr, "[host-pool-census] %s\n", s.str().c_str());

        std::vector<void *> seen;
        for (std::size_t i = 0; i < pool.block_count(); ++i)
        {
            seen.push_back(pool.addr(i));
        }
        // Blocks are distinct and stable for the pool's
        // life. A tile's blocks land CONTIGUOUS inside one segment
        // (the ladder's tile-wise act); adjacency ACROSS segments is
        // never promised — contig_bytes() is the truth.
        std::set<void *> distinct(seen.begin(), seen.end());
        CHECK(distinct.size() == want);

        // Writable at both ends of the carve (first byte of block 0,
        // last byte of the last block) — pinned host bytes.
        std::memcpy(pool.addr(0), "nnhaul", 7);
        CHECK(std::memcmp(pool.addr(0), "nnhaul", 7) == 0);
        char *last = static_cast<char *>(pool.addr(want - 1)) +
            chunk_class_bytes() - 1;
        *last = 42;
        CHECK(*last == 42);

        // Stability for the pool's life: the addresses did not move.
        for (std::size_t i = 0; i < want; ++i)
        {
            CHECK(pool.addr(i) == seen[i]);
        }

        bool threw = false;
        try
        {
            (void) pool.addr(want);
        }
        catch (std::runtime_error const &e)
        {
            threw = true;
            CHECK(std::string(e.what()).find("host pool") !=
                  std::string::npos);
        }
        CHECK(threw);
    } // the destructor releases the carve

    {
        // A second carve after the release, sized by precedence 2 this
        // time — release-then-recarve through the same one call.
        unset_env("NNHAUL_LIMIT_CPU_MEM");
        std::size_t const cap = std::size_t{32} << 20;
        CHECK(HostPool::resolve_limit_blocks(cap) ==
              cap / chunk_class_bytes());
        // Limit from precedence 2; the allocation ladder: a 3-block
        // ask lands EXACTLY 3 blocks — one TILE-SIZED segment for the
        // whole ask — never past the limit.
        HostPool pool(HostPool::resolve_limit_blocks(cap));
        std::size_t const got = pool.grow(3);
        CHECK(got == 3);
        CHECK(pool.block_count() == 3);
        CHECK(pool.addr(0) != nullptr);
        CHECK(pool.addr(2) != nullptr);
        CHECK(pool.segment_count() == 1);
        // Growth to the soft limit still serves the remainder, then
        // stops quietly. The remainder is class-dependent.
        CHECK(pool.grow(pool.limit_blocks()) == pool.limit_blocks() - 3);
        CHECK(pool.block_count() == pool.limit_blocks());
        CHECK(pool.grow(1) == 0); // at the limit: work within budget
    }

    // ---- the two-phase allocation ladder: the rule itself ----

    {
        // A scripted-failure pool: the test decides which allocation
        // acts fail, pinning the ladder's fallback and flip paths the
        // machine alone cannot be asked to produce.
        class FlakyPool : public nnhaul::HostPool
        {
          public:
            using HostPool::HostPool;
            bool fail_multi = false; // fail every multi-block act
            bool fail_all = false;   // fail every act (the flip)

          protected:
            bool allocate_segment(std::size_t n) override
            {
                if (fail_all || (fail_multi && n > 1))
                {
                    return false;
                }
                return HostPool::allocate_segment(n);
            }
        };

        // (1) Under a soft limit the episode fills to the WALL: the
        // tile event (3) plus the remaining-budget event (5) — 8/8,
        // still ALLOCABLE (no error happened).
        {
            HostPool pool(8);
            CHECK(pool.grow_to_wall(3) == 8);
            CHECK(pool.block_count() == 8);
            CHECK(!pool.growth_closed());
            CHECK(pool.grow_to_wall(1) == 0); // at the soft wall
            CHECK(!pool.growth_closed());     // the wall is not a flip
        }

        // (2) An UNLIMITED pool serves the demand only — the tile
        // event lands exactly the ask, nothing more.
        {
            HostPool pool(0);
            CHECK(pool.grow_to_wall(3) == 3);
            CHECK(pool.block_count() == 3);
            CHECK(pool.segment_count() == 1);
            CHECK(!pool.growth_closed());
        }

        // (3) Tile-wise failure -> SINGLE chunks for THIS event only;
        // success keeps the pool ALLOCABLE and the next event tries
        // tile-wise AGAIN (a fresh 2-block segment lands whole).
        {
            FlakyPool pool(8);
            pool.fail_multi = true;
            CHECK(pool.grow(3) == 3); // 3 singles, tile-wise failed
            CHECK(pool.block_count() == 3);
            CHECK(pool.segment_count() == 3);
            CHECK(!pool.growth_closed()); // stayed allocable
            pool.fail_multi = false;
            CHECK(pool.alloc_event(2) == 2); // tile-wise again — whole
            CHECK(pool.segment_count() == 4);
            CHECK(!pool.growth_closed());
        }

        // (4) A failed SINGLE-chunk allocation flips NON-ALLOCABLE,
        // permanently — even when the machine "heals" afterwards.
        {
            FlakyPool pool(8);
            CHECK(pool.grow(3) == 3);
            pool.fail_all = true;
            CHECK(pool.alloc_event(2) == 0); // tile-wise AND single
            CHECK(pool.growth_closed());
            pool.fail_all = false;          // memory healed...
            CHECK(pool.alloc_event(1) == 0); // ...non-allocable forever
            CHECK(pool.block_count() == 3);
            CHECK(pool.grow_to_wall(4) == 0);
        }
    }

    unset_env("NNHAUL_LIMIT_CPU_MEM");
    return g_checks_failed == 0 ? 0 : 1;
}
