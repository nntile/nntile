/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/nnhaul/chunk_gate.cc
 * The ONE process-wide chunked-tile allocation gate (see
 * include/nntile/nnhaul/chunk_gate.hh for the session law; out-of-line so
 * every shared library shares this single instance).
 *
 * @version 1.1.0
 * */

#include "nntile/nnhaul/chunk_gate.hh"

#ifdef NNTILE_USE_NNHAUL

#include <atomic>
#include <cstdlib>

namespace nntile::haul
{

namespace
{

std::atomic<bool> &tile_chunking_flag() noexcept
{
    // NNHAUL_CHUNK_TILES=0 starts the process with unchunked handles —
    // the opt-out knob Runtime::compile's torch-dispatch guard has
    // always advertised. STRICT parse: only a non-empty value that
    // fully parses as zero disables (strtol consumes the whole string);
    // unset, empty or non-numeric text keeps the chunked default, and
    // any nonzero value is an explicit pin to chunked. A loose atoi
    // would treat typos like "true" as 0 and silently switch every
    // allocation to contiguous. Read once at first use.
    static std::atomic<bool> flag{[]() noexcept {
        char const *e = std::getenv("NNHAUL_CHUNK_TILES");
        if(e == nullptr || *e == '\0')
        {
            return true;
        }
        char *end = nullptr;
        long const v = std::strtol(e, &end, 10);
        return !(*end == '\0' && v == 0);
    }()};
    return flag;
}

} // namespace

bool tile_chunking_enabled() noexcept
{
    return tile_chunking_flag().load(std::memory_order_relaxed);
}

void set_tile_chunking(bool on) noexcept
{
    tile_chunking_flag().store(on, std::memory_order_relaxed);
}

} // namespace nntile::haul

#endif // NNTILE_USE_NNHAUL
