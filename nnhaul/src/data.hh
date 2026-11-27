/*! @copyright (c) 2026-present Skolkovo Institute of Science and
 *                              Technology (Skoltech), Russia. All rights reserved.
 *
 * NNHaul is a task-based dynamic execution engine over N memory nodes:
 * a virtual planner builds one plan per batch from a drained snapshot,
 * and a literal executor issues exactly the planned operations on the
 * lanes of the node inventory.
 *
 * @file src/data.hh
 * HandleImpl: the TILE-SIDE facts only (nbytes, n, name). Residency
 * is NOT here — the binding table owns it; the host pool owns the
 * host node's bytes. The per-chunk geometry is derived from nbytes
 * through the one chunk-class rule, never a per-handle declaration.
 *
 * @version 1.0.0
 * */

// SPDX-License-Identifier: MIT

#pragma once

// HandleImpl under the chunk-graph engine. The previous engine's
// per-handle residency machinery — the device replica, the host
// shadow page, presence facts, the host-dirty reseed bit — is deleted:
// a tile's bytes are whole-class chunks that the one binding table
// binds on the nodes they occupy, and the only per-handle state left
// is the tile's own facts plus the host-pin count the Acquired surface
// uses. The chunk geometry is derived here: a tile of nbytes occupies
// n = ceil(nbytes / chunk_class_bytes()) whole-class chunks, the last
// chunk's remainder being the intra-chunk tail; the codelet's chunk
// accessors iterate exactly that, and the engine's launcher maps chunk
// i to plan chunk i — identity.

#include "nnhaul/nnhaul.hh"
#include "plan.hh"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace nnhaul
{

struct HandleImpl
{
    // The tile-side facts: the payload size, the diagnostic
    // name, and the chunk count derived from the size.
    std::size_t nbytes = 0;
    std::string name;

    // The one-class decomposition:
    // n = tile_chunk_count(nbytes) — ceil(nbytes / chunk_class_bytes()),
    // every chunk a whole class block; the last chunk's remainder is
    // the intra-chunk tail, wasted bytes inside the block. A tile at
    // or below the class is exactly ONE chunk and presents the
    // contiguous get_ptr view; every larger tile presents the
    // per-chunk view. There is no declared per-handle geometry, no
    // tail repacking, and no sharing: chunk i of the tile is a whole
    // class slot of this tile alone.
    std::size_t n = 0;

    // Sets the tile facts and derives n (the constructor the Handle
    // constructors go through; the one-class rule, nothing else).
    explicit HandleImpl(std::size_t nbytes_, std::string name_ = "")
        : nbytes(nbytes_),
          name(std::move(name_)),
          n(tile_chunk_count(nbytes))
    {
    }

    HandleImpl() = default;

    // Logical payload bytes of chunk i — tile_chunk_span's arithmetic
    // (plan.cc, the single definition:
    // min(chunk_class_bytes(), nbytes - i * chunk_class_bytes())).
    std::size_t chunk_nbytes(std::size_t i) const
    {
        return tile_chunk_span(nbytes, static_cast<ChunkIndex>(i));
    }

    bool is_chunked() const { return n > 1; }

    // The tile's plan-world identity: the creation-order TileId that
    // the binding table's logical chunks name. Assigned at handle
    // construction; stable for the handle's life.
    TileId tile_id = 0;

    //! The Handle object is gone but retirement has not run yet; the
    //! engine owns this allocation and deletes it at the sweep.
    bool engine_owned = false;
    //! This impl is already in an un-retired batch's sweep list.
    bool swept_pending = false;

    // The public invalidate surface (Handle::invalidate): dead marks
    // the marker; dead_done once the marker is joined and the tile's
    // bindings unlinked (handle_swept's fact).
    std::atomic<bool> dead{false};
    bool dead_done = false;

    // The host API's pin count: live Acquired instances (the Acquired
    // surface; the engine wires the pin/unpin acts).
    int pins = 0;

    // The O(1) handle bookkeeping: the dead-pending and registry links
    // live IN the handle — an intrusive doubly-linked list per
    // structure — so listing and delisting are pointer surgery on
    // these fields, never a container scan. dead_* run under the
    // runtime state's mu, reg_* under the api's registry mutex; null
    // prev and a non-head identity means "not listed".
    HandleImpl *dead_prev = nullptr;
    HandleImpl *dead_next = nullptr;
    HandleImpl *reg_prev = nullptr;
    HandleImpl *reg_next = nullptr;
};

inline std::string label_of(HandleImpl const &impl)
{
    return impl.name.empty() ? std::string("unnamed") : impl.name;
}

} // namespace nnhaul
