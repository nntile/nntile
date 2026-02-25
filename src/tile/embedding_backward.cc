/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/embedding_backward.cc
 * Backward embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/embedding_backward.hh"
#include "nntile/starpu/embedding_backward.hh"
#include "nntile/starpu/config.hh"

namespace nntile::tile
{

template<typename T>
void embedding_backward_async(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<T> &embed,
        const Tile<T> &vocab, int redux)
{
    int mpi_rank = starpu_mpi_world_rank();
    int vocab_rank = vocab.mpi_get_rank();
    index.mpi_transfer(vocab_rank, mpi_rank);
    embed.mpi_transfer(vocab_rank, mpi_rank);
    if(mpi_rank == vocab_rank)
    {
        starpu::embedding_backward.submit<std::tuple<T>>(m, n, k, k_start,
                k_size, index, embed, vocab, redux);
    }
}

template<typename T>
void embedding_backward(Index m, Index n, Index k, Index k_start, Index k_size,
        const Tile<int64_t> &index, const Tile<T> &embed,
        const Tile<T> &vocab, int redux)
{
    embedding_backward_async<T>(m, n, k, k_start, k_size, index, embed, vocab,
            redux);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void embedding_backward_async<fp32_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp32_t> &embed,
        const Tile<fp32_t> &vocab, int redux);

template
void embedding_backward_async<fp32_fast_tf32_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &embed,
        const Tile<fp32_fast_tf32_t> &vocab, int redux);

template
void embedding_backward_async<fp32_fast_fp16_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &embed,
        const Tile<fp32_fast_fp16_t> &vocab, int redux);

template
void embedding_backward_async<fp32_fast_bf16_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &embed,
        const Tile<fp32_fast_bf16_t> &vocab, int redux);

template
void embedding_backward_async<fp64_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp64_t> &embed,
        const Tile<fp64_t> &vocab, int redux);

template
void embedding_backward_async<bf16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<bf16_t> &embed,
        const Tile<bf16_t> &vocab, int redux);

template
void embedding_backward_async<fp16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp16_t> &embed,
        const Tile<fp16_t> &vocab, int redux);

// Explicit instantiation
template
void embedding_backward<fp32_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp32_t> &embed,
        const Tile<fp32_t> &vocab, int redux);

template
void embedding_backward<fp32_fast_tf32_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_tf32_t> &embed,
        const Tile<fp32_fast_tf32_t> &vocab, int redux);

template
void embedding_backward<fp32_fast_fp16_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_fp16_t> &embed,
        const Tile<fp32_fast_fp16_t> &vocab, int redux);

template
void embedding_backward<fp32_fast_bf16_t>(Index m, Index n, Index k,
        Index k_start, Index k_size, const Tile<int64_t> &index,
        const Tile<fp32_fast_bf16_t> &embed,
        const Tile<fp32_fast_bf16_t> &vocab, int redux);

template
void embedding_backward<fp64_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp64_t> &embed,
        const Tile<fp64_t> &vocab, int redux);

template
void embedding_backward<bf16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<bf16_t> &embed,
        const Tile<bf16_t> &vocab, int redux);

template
void embedding_backward<fp16_t>(Index m, Index n, Index k, Index k_start,
        Index k_size, const Tile<int64_t> &index, const Tile<fp16_t> &embed,
        const Tile<fp16_t> &vocab, int redux);

} // namespace nntile::tile
