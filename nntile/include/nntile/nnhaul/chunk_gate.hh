/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/nnhaul/chunk_gate.hh
 * Process-wide gate for chunked tile allocation in the NNHaul build.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_USE_NNHAUL

#include <atomic>

namespace nntile::haul
{

/*! Whether newly allocated tiles may use chunked NNHaul handles.
 *
 * Chunked tiles exist only for the classic (non torch_dispatch) stack:
 * torch-native ops consume tiles through contiguous torch tensors and
 * cannot read a chunk chain. Runtime::compile() rejects the forbidden
 * combination loudly - it throws before execution when the graph lowers
 * any TILE_TORCH_* op while chunked tile handles are still allocated -
 * and only then clears this gate so every tile allocated afterwards is
 * a plain contiguous handle (the gate affects FUTURE allocations only;
 * that is why the throw, not a silent flip, is the guard). The gate is
 * process-wide on purpose: torch_nntile mixes classic and torch ops on
 * the same tiles within one session, so chunking must be off for all
 * graphs of that session. The env knob NNHAUL_CHUNK_TILES=0 forces
 * unchunked handles (the one-class law has no such arm) regardless of
 * this gate.
 * */
inline std::atomic<bool> &tile_chunking_flag() noexcept
{
    static std::atomic<bool> flag{true};
    return flag;
}

inline bool tile_chunking_enabled() noexcept
{
    return tile_chunking_flag().load(std::memory_order_relaxed);
}

//! Set the chunked-tile allocation gate (see tile_chunking_enabled()).
inline void set_tile_chunking(bool on) noexcept
{
    tile_chunking_flag().store(on, std::memory_order_relaxed);
}

} // namespace nntile::haul

#endif // NNTILE_USE_NNHAUL
