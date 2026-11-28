/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/embedding.cc
 * Embeddings from vocabulary within Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/embedding.hh"
#include "nntile/nnhaul/ops/embedding.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void embedding_async(int worker_hint, Index m, Index n, Index k, Index k_start, Index k_size, Index index_range,
        const Tile<int64_t> &index, const Tile<T> &vocab,
        const Tile<T> &embed)
{
    nnhaul::embedding.submit<std::tuple<T>>(worker_hint, m, n, k, k_start, k_size,
            index_range, index, vocab, embed);
}

template<typename T>
void embedding(int worker_hint, Index m, Index n, Index k, Index k_start, Index k_size, Index index_range,
        const Tile<int64_t> &index, const Tile<T> &vocab,
        const Tile<T> &embed)
{
    embedding_async<T>(worker_hint, m, n, k, k_start, k_size,
            index_range, index, vocab, embed);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void embedding_async<fp32_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index, const Tile<fp32_t> &vocab,
        const Tile<fp32_t> &embed);

template
void embedding_async<bf16_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index, const Tile<bf16_t> &vocab,
        const Tile<bf16_t> &embed);

template
void embedding_async<fp16_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index, const Tile<fp16_t> &vocab,
        const Tile<fp16_t> &embed);

template
void embedding_async<fp32_fast_tf32_t>(int worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &vocab,
        const Tile<fp32_fast_tf32_t> &embed);

template
void embedding_async<fp32_fast_fp16_t>(int worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &vocab,
        const Tile<fp32_fast_fp16_t> &embed);

template
void embedding_async<fp32_fast_bf16_t>(int worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &vocab,
        const Tile<fp32_fast_bf16_t> &embed);

template
void embedding_async<fp64_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index, const Tile<fp64_t> &vocab,
        const Tile<fp64_t> &embed);

// Explicit instantiation
template
void embedding<fp32_t>(int worker_hint, Index m, Index n, Index k, Index k_start, Index k_size, Index index_range,
        const Tile<int64_t> &index, const Tile<fp32_t> &vocab,
        const Tile<fp32_t> &embed);

template
void embedding<bf16_t>(int worker_hint, Index m, Index n, Index k, Index k_start, Index k_size, Index index_range,
        const Tile<int64_t> &index, const Tile<bf16_t> &vocab,
        const Tile<bf16_t> &embed);

template
void embedding<fp16_t>(int worker_hint, Index m, Index n, Index k, Index k_start, Index k_size, Index index_range,
        const Tile<int64_t> &index, const Tile<fp16_t> &vocab,
        const Tile<fp16_t> &embed);

template
void embedding<fp32_fast_tf32_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &vocab,
        const Tile<fp32_fast_tf32_t> &embed);

template
void embedding<fp32_fast_fp16_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &vocab,
        const Tile<fp32_fast_fp16_t> &embed);

template
void embedding<fp32_fast_bf16_t>(int worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, Index index_range, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &vocab,
        const Tile<fp32_fast_bf16_t> &embed);

template
void embedding<fp64_t>(int worker_hint, Index m, Index n, Index k, Index k_start, Index k_size, Index index_range,
        const Tile<int64_t> &index, const Tile<fp64_t> &vocab,
        const Tile<fp64_t> &embed);

} // namespace nntile::core
