/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/kernel/embedding_backward/cuda.hh
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/base_types.hh>
#include <cuda_runtime.h>

namespace nntile::kernel::embedding_backward
{

//! vocab = beta*vocab + alpha*scatter(embed); beta in {0,1}
/*! If beta=0, the entire vocab buffer (vocab_nelems) is zeroed first, then
 * indexed rows accumulate alpha*embed. If beta=1, only accumulate.
 */
template<typename T>
void cuda(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, Index vocab_nelems, Scalar alpha,
        Scalar beta, const int64_t *index, const T *embed, T *vocab)
    noexcept;

//! vocab = beta*vocab + alpha*scatter(embed) on a chunked vocabulary
/*! Same operation as cuda() above, but each operand's flat element range
 * is backed by a chain of equal-size chunk allocations (a shorter tail on
 * the last chunk). The whole scatter-add is issued as ONE atomicAdd kernel
 * launch whose chunk base-pointer tables are read from the device
 * chunk-pointer ring (no per-tile or handle cap); each element derives its
 * chunk index from its own flat offset (chunk = offset / elems, in-chunk
 * offset = offset % elems). The beta==0 zeroing is a single chunk-aware
 * kernel launch as well. The accumulation is atomicAdd exactly as the
 * whole-buffer entry's, so the result is the same unordered sum (and
 * bitwise the whole-buffer call when every token is unique). A launch
 * whose pointer set exceeds the ring falls back to the per-segment driver.
 *
 * @param[in] stream: CUDA stream
 * @param[in] m: Size of the first mode of index and embed tensors
 * @param[in] n: Size of the last mode of index and embed tensors
 * @param[in] k: Size of the middle mode of embed tensor
 * @param[in] k_start: Offset of the middle mode of embed tensor
 * @param[in] k_size: Size of the first mode of vocab tensor
 * @param[in] index_range: Number of rows of vocab (valid token ids)
 * @param[in] vocab_nelems: Total elements of the vocab buffer (for beta=0)
 * @param[in] alpha: Scalar multiplier of the contribution
 * @param[in] beta: 0 to zero vocab first, 1 to accumulate only
 * @param[in] index_chunks: Device-ring view of index's chunk base pointers
 * @param[in] nindex: Number of index chunks
 * @param[in] elems_index: Elements in a full index chunk (uniform)
 * @param[in] embed_chunks: Device-ring view of embed's chunk base pointers
 * @param[in] nembed: Number of embed chunks
 * @param[in] elems_embed: Elements in a full embed chunk (uniform)
 * @param[in] vocab_chunks: Device-ring view of vocab's chunk base pointers
 * @param[in] nvocab: Number of vocab chunks
 * @param[in] elems_vocab: Elements in a full vocab chunk (uniform)
 * */
template<typename T>
void chunked(cudaStream_t stream, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, Index vocab_nelems, Scalar alpha,
        Scalar beta,
        const int64_t *const *index_chunks, Index nindex, Index elems_index,
        const T *const *embed_chunks, Index nembed, Index elems_embed,
        T *const *vocab_chunks, Index nvocab, Index elems_vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
