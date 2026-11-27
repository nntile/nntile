/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on the NNHaul runtime system.
 *
 * @file include/nntile/nnhaul/chunk_gate.hh
 * Process-wide gate for chunked tile allocation in the NNHaul build.
 *
 * @version 1.1.0
 * */

#pragma once


namespace nntile::nnhaul
{

/*! Whether newly allocated tiles may use chunked NNHaul handles.
 *
 * On the NNHaul backend the answer is always yes: every tile is
 * chunk-backed. The gate exists so the pipeline can state that rule in
 * one place and so a mixed binary fails closed.
 *
 * The environment knob is strict:
 *   - unset, empty or `1`: chunked (the default);
 *   - `0`: rejected loudly, because a contiguous tile handle is exactly
 *     the unchunked launch path this backend forbids;
 *   - any other value: chunked.
 *
 * Runtime::compile() still throws if a graph lowers TILE_TORCH_* while
 * chunked handles are alive. That op family is not compiled into this
 * product, so the throw only catches a stale or hand-built object; it
 * never demotes the gate to make such a graph run.
 *
 * Out-of-line ON PURPOSE (defined in nnhaul/chunk_gate.cc): the flag
 * must be ONE process-wide instance. An inline function-local static
 * gives every shared library that inlines it its own copy - a flip
 * inside torch_nntile's extension module would then never reach the
 * copy Tile::alloc_handle reads in libnntile.
 * */
bool tile_chunking_enabled();

} // namespace nntile::nnhaul
