/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file src/nnhaul/stage.cc
 * State of the chunked-handle bookkeeping: the chunked-handle registry.
 * The former staged fallback and its scratch queue no longer exist.
 *
 * @version 2.0.0
 * */

#include "nntile/nnhaul/stage.hh"

#include <mutex>
#include <unordered_map>

namespace nntile
{
namespace nnhaul
{
namespace stage
{

namespace
{

std::mutex g_mu;
// Chunked handles whose producer announced them to NNTile, with the
// chunk size in bytes when the producer knew it (0 = unknown).
std::unordered_map<::nnhaul::Handle *, std::size_t> g_chunked;

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
    // Unknown producer: Tile::alloc_handle's decision, replayed. One
    // chunk class (the new nnhaul engine): every tile decomposes into
    // whole chunk-class chunks, so chunkedness is the gate.
    return ::nntile::nnhaul::tile_chunking_enabled();
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
    // One chunk class (the new nnhaul engine): the class itself.
    return ::nnhaul::chunk_class_bytes();
}

std::size_t chunked_handle_count()
{
    std::lock_guard<std::mutex> lock(g_mu);
    return g_chunked.size();
}

} // namespace stage
} // namespace nnhaul
} // namespace nntile
