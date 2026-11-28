/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/stage.hh
 * Chunked-handle bookkeeping for classic NNTile ops on NNHaul tiles: the
 * chunked-handle registry used by submit-time geometry guards.
 *
 * There is no staged fallback: the former gather/run/scatter machinery was
 * removed once every product driver got a chunk-aware path. Drivers use
 * chunked kernels or chunk-aware piecewise drivers.
 *
 * @version 2.0.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <nntile/nnhaul/chunk_gate.hh>
#include <nntile/nnhaul/codelet.hh>

#include <nnhaul/nnhaul.hh>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace nntile
{
namespace nnhaul
{
namespace stage
{

// ---- chunked-handle detection ------------------------------------------
//
// NNHaul's public Handle API exposes no chunk query at submit time (the
// chunk table only materializes on the worker's Buffer snapshot inside a
// kernel), so detection reconstructs the allocation decision from two
// sources:
//
// 1. The registry (exact): every chunked handle whose producer is known
//    to NNTile registers itself here - core::Tile::alloc_handle for
//    graph tiles, note_chunked() for raw two-argument handles built by
//    embedders and tests outside the Tile allocation path.
// 2. The policy recompute (conservative): an unknown handle is treated
//    as chunked when the classic gate is open and the nnhaul ladder
//    would chunk its nbytes. This is exactly Tile::alloc_handle's
//    decision, so it is exact for tile payloads.

//! Remember a chunked handle for handle_is_chunked() and the compile-time
//! torch gate. Idempotent. Call before submitting tasks that use it.
//! chunk_bytes (when the producer knows it - Tile::alloc_handle and the
//! test factories do) is recorded for handle_chunk_bytes(): submit-time
//! launch-count guards need the handle's REAL chunk class, which the
//! policy recompute can only approximate for foreign handles.
void note_chunked(::nnhaul::Handle &handle,
    std::size_t chunk_bytes = 0);

//! Drop a remembered handle (idempotent). Must run before the handle is
//! destroyed so a recycled address cannot alias a stale entry.
void forget_handle(::nnhaul::Handle &handle);

//! True when a contiguous-pointer codelet must treat this handle as
//! chunked (it then has to use the chunk-aware path).
bool handle_is_chunked(::nnhaul::Handle &handle);

//! The handle's chunk size in bytes: the recorded value for announced
//! handles, the policy recompute (Tile::alloc_handle's exact decision)
//! for unknown ones, 0 when the handle is not chunked. Submit-time
//! geometry guards (transpose's degenerate-panel guard) read this.
std::size_t handle_chunk_bytes(::nnhaul::Handle &handle);

//! Number of live remembered chunked handles (the torch-op gate reads
//! this at compile time).
std::size_t chunked_handle_count();

//! Engagement probe for the removed universal staged fallback: no submit
//! takes that path any more, so the counter must stay zero. The chunked-tile
//! tests assert exactly that (they used to prove the fallback never engaged).
inline std::atomic<std::uint64_t> g_staged_submit_count{0};

inline void note_staged_submit() noexcept
{
    g_staged_submit_count.fetch_add(1, std::memory_order_relaxed);
}

inline std::uint64_t staged_submit_count() noexcept
{
    return g_staged_submit_count.load(std::memory_order_relaxed);
}

} // namespace stage
} // namespace nnhaul
} // namespace nntile
