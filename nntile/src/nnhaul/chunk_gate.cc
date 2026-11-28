/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file src/nnhaul/chunk_gate.cc
 * The ONE process-wide chunked-tile allocation gate (see
 * include/nntile/nnhaul/chunk_gate.hh for the session rule; out-of-line so
 * every shared library shares this single instance).
 *
 * @version 2.0.0
 * */

#include "nntile/nnhaul/chunk_gate.hh"

#include <atomic>
#include <cstdlib>
#include <stdexcept>

namespace nntile::nnhaul
{

namespace
{

std::atomic<bool> &tile_chunking_flag()
{
    // Every tile is chunk-backed on this backend; the 1.x switch (including
    // the NNHAUL_CHUNK_TILES=0 opt-out and any nonzero "pin") no longer
    // selects anything. Reject any value instead of accepting it as a
    // silent no-op.
    static std::atomic<bool> flag{[]() {
        char const *e = std::getenv("NNHAUL_CHUNK_TILES");
        if(e == nullptr || *e == '\0')
        {
            return true;
        }
        throw std::runtime_error(
            "nntile: NNHAUL_CHUNK_TILES is not supported in NNTile 2.0: "
            "every tile is chunk-backed and there is no contiguous "
            "allocation mode. Unset the variable.");
    }()};
    return flag;
}

} // namespace

bool tile_chunking_enabled()
{
    return tile_chunking_flag().load(std::memory_order_relaxed);
}

} // namespace nntile::nnhaul
