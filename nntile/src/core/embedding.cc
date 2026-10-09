/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/embedding.cc
 * Embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/embedding.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/embedding.hh"
#else
#include "nntile/starpu/embedding.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

template<typename T>
void embedding_async(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<T> &vocab,
        const Tile<T> &embed)
{
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int embed_rank = 0;
    #else
    int embed_rank = embed.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    index.mpi_transfer(embed_rank, mpi_rank);
    #endif
    #ifndef NNTILE_USE_NNHAUL
    vocab.mpi_transfer(embed_rank, mpi_rank);
    #endif
    if(mpi_rank == embed_rank)
    {
        #ifdef NNTILE_USE_NNHAUL
        haul::embedding.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, k_start, k_size,
                index, vocab, embed);
        #else
        starpu::embedding.submit<std::tuple<T>>(starpu_worker_hint, m, n, k, k_start, k_size,
                index, vocab, embed);
        #endif

    }
}

template<typename T>
void embedding(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<T> &vocab,
        const Tile<T> &embed)
{
    embedding_async<T>(starpu_worker_hint, m, n, k, k_start, k_size, index, vocab, embed);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void embedding_async<fp32_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp32_t> &vocab,
        const Tile<fp32_t> &embed);

template
void embedding_async<bf16_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<bf16_t> &vocab,
        const Tile<bf16_t> &embed);

template
void embedding_async<fp16_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp16_t> &vocab,
        const Tile<fp16_t> &embed);

template
void embedding_async<fp32_fast_tf32_t>(int starpu_worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &vocab,
        const Tile<fp32_fast_tf32_t> &embed);

template
void embedding_async<fp32_fast_fp16_t>(int starpu_worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &vocab,
        const Tile<fp32_fast_fp16_t> &embed);

template
void embedding_async<fp32_fast_bf16_t>(int starpu_worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &vocab,
        const Tile<fp32_fast_bf16_t> &embed);

template
void embedding_async<fp64_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp64_t> &vocab,
        const Tile<fp64_t> &embed);

// Explicit instantiation
template
void embedding<fp32_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<fp32_t> &vocab,
        const Tile<fp32_t> &embed);

template
void embedding<bf16_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<bf16_t> &vocab,
        const Tile<bf16_t> &embed);

template
void embedding<fp16_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<fp16_t> &vocab,
        const Tile<fp16_t> &embed);

template
void embedding<fp32_fast_tf32_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &vocab,
        const Tile<fp32_fast_tf32_t> &embed);

template
void embedding<fp32_fast_fp16_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &vocab,
        const Tile<fp32_fast_fp16_t> &embed);

template
void embedding<fp32_fast_bf16_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &vocab,
        const Tile<fp32_fast_bf16_t> &embed);

template
void embedding<fp64_t>(int starpu_worker_hint, Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<fp64_t> &vocab,
        const Tile<fp64_t> &embed);

} // namespace nntile::core
