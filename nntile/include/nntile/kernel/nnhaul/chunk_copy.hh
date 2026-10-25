/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/nnhaul/chunk_copy.hh
 * Single-launch chunk gather/scatter kernels over NNHaul chunk tables.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/defs.h>

#if defined(NNTILE_USE_CUDA) && defined(NNTILE_USE_NNHAUL)

#include <nnhaul/nnhaul.hh>

#include <cstddef>

namespace nntile::kernel::nnhaul
{

/*! Gather all chunks of `src` into one contiguous destination with a
 * SINGLE device kernel launch (in-codelet only): the destination is
 * split into class-sized chunks implicitly; a small descriptor table
 * (one entry per source chunk) is uploaded to a preallocated device
 * scratch and one kernel walks it. Equivalent to copying chunk by
 * chunk, without one API call per chunk. */
void gather_chunks(::nnhaul::Buffer const &src, void *dst_contiguous);

/*! Scatter one contiguous source into all chunks of `dst` with a
 * single device kernel launch (the inverse of gather_chunks). */
void scatter_chunks(void const *src_contiguous, ::nnhaul::Buffer &dst);

} // namespace nntile::kernel::nnhaul

#endif // NNTILE_USE_CUDA && NNTILE_USE_NNHAUL
