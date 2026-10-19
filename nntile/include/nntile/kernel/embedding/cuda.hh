/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding/cuda.hh
 * Embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::embedding
{

// Fill embedding from vocabulary
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const int64_t *index,
        const T *vocab, T *embed)
    noexcept;

//! Fill embedding from a chunked vocabulary on CUDA
/*! Same operation as cuda() above, but each operand's flat element range
 * is backed by a chain of equal-size chunk allocations (a shorter tail on
 * the last chunk). The whole gather is issued as ONE kernel launch: the
 * three chunk base-pointer tables are read from the device chunk-pointer
 * ring (no per-tile or handle cap), and each element derives its chunk
 * index from its own flat offset (chunk = offset / elems, in-chunk offset
 * = offset % elems). A pure copy, so the result is bitwise the
 * whole-buffer cuda() call. A launch whose pointer set exceeds the ring
 * falls back to the per-panel/segment driver.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index_range: Number of rows of vocab (valid token ids)
 * @param[in] index_chunks: Device-ring view of index's chunk base pointers
 * @param[in] nindex: Number of index chunks
 * @param[in] elems_index: Elements in a full index chunk (uniform)
 * @param[in] vocab_chunks: Device-ring view of vocab's chunk base pointers
 * @param[in] nvocab: Number of vocab chunks
 * @param[in] elems_vocab: Elements in a full vocab chunk (uniform)
 * @param[in] embed_chunks: Device-ring view of embed's chunk base pointers
 * @param[in] nembed: Number of embed chunks
 * @param[in] elems_embed: Elements in a full embed chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const T *const *vocab_chunks, Index nvocab, Index elems_vocab,
        T *const *embed_chunks, Index nembed, Index elems_embed)
    noexcept;

} // namespace nntile::kernel::embedding
