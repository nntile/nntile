/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/nnhaul/stage.cc
 * State of the universal staging fallback: the chunked-handle registry
 * and the live scratch-handle set (see nnhaul/stage.hh for the lifetime
 * scheme).
 *
 * @version 1.1.0
 * */

#include "nntile/nnhaul/stage.hh"

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nntile
{
namespace haul
{
namespace stage
{

namespace
{

std::mutex g_mu;
// Chunked handles whose producer announced them to NNTile, with the
// chunk size in bytes when the producer knew it (0 = unknown).
std::unordered_map<::nnhaul::Handle *, std::size_t> g_chunked;
// Every scratch handle staged since the last idle drain. Destroyed
// ONLY by drain_scratches() - when the task queue is known idle (or
// after nnhaul::shutdown()); see the lifetime scheme in stage.hh.
std::vector<std::unique_ptr<::nnhaul::Handle>> g_live;

} // namespace

void note_chunked(::nnhaul::Handle &handle, std::size_t chunk_bytes)
{
    std::lock_guard<std::mutex> lock(g_mu);
    g_chunked[&handle] = chunk_bytes;
}

void forget_handle(::nnhaul::Handle &handle)
{
    std::lock_guard<std::mutex> lock(g_mu);
    g_chunked.erase(&handle);
}

bool handle_is_chunked(::nnhaul::Handle &handle)
{
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_chunked.count(&handle) != 0)
        {
            return true;
        }
    }
    // Unknown producer: Tile::alloc_handle's decision, replayed. Erring
    // towards staging here is correct (staging also works for unchunked
    // buffers); Runtime::compile guarantees no chunked tile payload is
    // alive while the gate is off, so this cannot miss a live tile.
    return ::nntile::haul::tile_chunking_enabled()
        && ::nnhaul::chunk_bytes_for(handle.nbytes()) != 0;
}

std::size_t handle_chunk_bytes(::nnhaul::Handle &handle)
{
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto const at = g_chunked.find(&handle);
        if (at != g_chunked.end())
        {
            // Announced: the recorded size (0 only when the producer
            // did not know it - fall through to the recompute).
            if(at->second != 0)
            {
                return at->second;
            }
        }
    }
    if(!handle_is_chunked(handle))
    {
        return 0;
    }
    // Tile::alloc_handle's own chunk decision, replayed (exact for tile
    // payloads; the best available estimate for foreign handles).
    return ::nnhaul::chunk_bytes_for(handle.nbytes());
}

std::size_t chunked_handle_count()
{
    std::lock_guard<std::mutex> lock(g_mu);
    return g_chunked.size();
}

void stash_scratches(
    std::vector<std::unique_ptr<::nnhaul::Handle>> scratches)
{
    // The live set spans every staged submit since the last idle drain
    // (see the lifetime scheme in stage.hh: handles are destroyed only
    // when the queue is idle). Appending is enough; the set is
    // destroyed whole by drain_scratches().
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &h : scratches)
    {
        g_live.push_back(std::move(h));
    }
}

void drain_scratches()
{
    std::vector<std::unique_ptr<::nnhaul::Handle>> old;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        old.swap(g_live);
    }
    // The queue is idle (the caller just returned from nnhaul::wait()):
    // each scratch's invalidate marker sweeps immediately, so these
    // destructors do not block. After nnhaul::shutdown() (the exit
    // fallback) no runtime exists and the destructors are trivial
    // frees. Either way nothing can still reference the handles.
    old.clear();
}

} // namespace stage
} // namespace haul
} // namespace nntile
