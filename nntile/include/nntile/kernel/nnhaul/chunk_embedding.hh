/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/nnhaul/chunk_embedding.hh
 * Device-side embedding gather/scatter kernels over NNHaul chunk tables.
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

/*! Device-side embedding gather over chunk tables (in-codelet only):
 * embed[i0, k_start+i2, i1] = vocab[k_size*index[i0 + m*i1] + i2].
 * The token index stays on the device — no D2H, no stream sync. One
 * launch. elem_size selects the element type (4 or 8 bytes). */
void embedding_gather(::nnhaul::Buffer const &index,
    ::nnhaul::Buffer const &vocab, ::nnhaul::Buffer &embed,
    std::size_t m, std::size_t n, std::size_t k, std::size_t k_start,
    std::size_t k_size, std::size_t index_range, std::size_t elem_size);

/*! Device-side embedding backward over chunk tables (in-codelet only):
 * vocab[k_size*index[..] + i2] += alpha * embed[...]; beta == 0 zeroes
 * the vocabulary first. The index stays on the device. One launch. */
void embedding_backward_add(::nnhaul::Buffer const &index,
    ::nnhaul::Buffer const &embed, ::nnhaul::Buffer &vocab,
    std::size_t m, std::size_t n, std::size_t k, std::size_t k_start,
    std::size_t k_size, std::size_t index_range, double alpha, double beta,
    std::size_t elem_size);

} // namespace nntile::kernel::nnhaul

#endif // NNTILE_USE_CUDA && NNTILE_USE_NNHAUL
