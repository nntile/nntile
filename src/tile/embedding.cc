/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/embedding.cc
 * Embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/embedding.hh"
#include "nntile/starpu/embedding.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void embedding_async(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<T> &vocab,
        const Tile<T> &embed)
{
    int mpi_rank = starpu_mpi_world_rank();
    int embed_rank = embed.mpi_get_rank();
    index.mpi_transfer(embed_rank, mpi_rank);
    vocab.mpi_transfer(embed_rank, mpi_rank);
    if(mpi_rank == embed_rank)
    {
        starpu::embedding.submit<std::tuple<T>>(m, n, k, k_start, k_size,
                index, vocab, embed);
    }
}

template<typename T>
void embedding(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<T> &vocab,
        const Tile<T> &embed)
{
    embedding_async<T>(m, n, k, k_start, k_size, index, vocab, embed);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void embedding_async<fp32_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp32_t> &vocab,
        const Tile<fp32_t> &embed);

template
void embedding_async<bf16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<bf16_t> &vocab,
        const Tile<bf16_t> &embed);

template
void embedding_async<fp16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp16_t> &vocab,
        const Tile<fp16_t> &embed);

template
void embedding_async<fp32_fast_tf32_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &vocab,
        const Tile<fp32_fast_tf32_t> &embed);

template
void embedding_async<fp32_fast_fp16_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &vocab,
        const Tile<fp32_fast_fp16_t> &embed);

template
void embedding_async<fp32_fast_bf16_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &vocab,
        const Tile<fp32_fast_bf16_t> &embed);

template
void embedding_async<fp64_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp64_t> &vocab,
        const Tile<fp64_t> &embed);

// Explicit instantiation
template
void embedding<fp32_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<fp32_t> &vocab,
        const Tile<fp32_t> &embed);

template
void embedding<bf16_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<bf16_t> &vocab,
        const Tile<bf16_t> &embed);

template
void embedding<fp16_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<fp16_t> &vocab,
        const Tile<fp16_t> &embed);

template
void embedding<fp32_fast_tf32_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &vocab,
        const Tile<fp32_fast_tf32_t> &embed);

template
void embedding<fp32_fast_fp16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &vocab,
        const Tile<fp32_fast_fp16_t> &embed);

template
void embedding<fp32_fast_bf16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &vocab,
        const Tile<fp32_fast_bf16_t> &embed);

template
void embedding<fp64_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<fp64_t> &vocab,
        const Tile<fp64_t> &embed);

} // namespace nntile::tile
