/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/nnhaul/chunk_gate.hh
 * Process-wide gate for chunked tile allocation in the NNHaul build.
 *
 * @version 1.1.0
 * */

#pragma once

#ifdef NNTILE_USE_NNHAUL

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
 * graphs of that session. The law is ONE-DIRECTIONAL at runtime: a
 * torch-op graph (or the torch_nntile session policy) turns chunking
 * off for the rest of the process, and nothing turns it back on.
 * The env knob NNHAUL_CHUNK_TILES initializes the gate instead:
 * =0 (strictly parsed, whole value) starts the process with unchunked
 * handles; any nonzero value pins chunked handles on; unset keeps the
 * chunked default for classic sessions and the torch_nntile session
 * policy decides for torch sessions.
 *
 * Out-of-line ON PURPOSE (defined in nnhaul/chunk_gate.cc): the flag
 * must be ONE process-wide instance. An inline function-local static
 * gives every shared library that inlines it its own copy - a flip
 * inside torch_nntile's extension module would then never reach the
 * copy Tile::alloc_handle reads in libnntile.
 * */
bool tile_chunking_enabled() noexcept;

//! Set the chunked-tile allocation gate (see tile_chunking_enabled()).
void set_tile_chunking(bool on) noexcept;

} // namespace nntile::haul

#endif // NNTILE_USE_NNHAUL
