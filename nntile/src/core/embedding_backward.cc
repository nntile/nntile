/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/embedding_backward.cc
 * Backward embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/embedding_backward.hh"
#include "nntile/starpu/embedding_backward.hh"
#include "nntile/starpu/config.hh"

namespace nntile::core
{

template<typename T>
void embedding_backward_async(int starpu_worker_hint, Index m, Index n,
        Index k, Index k_start, Index k_size, Scalar alpha, Scalar beta,
        const Tile<int64_t> &index, const Tile<T> &embed, const Tile<T> &vocab,
        int redux)
{
    int mpi_rank = starpu_mpi_world_rank();
    int vocab_rank = vocab.mpi_get_rank();
    index.mpi_transfer(vocab_rank, mpi_rank);
    embed.mpi_transfer(vocab_rank, mpi_rank);
    if(mpi_rank == vocab_rank)
    {
        starpu::embedding_backward.submit<std::tuple<T>>(starpu_worker_hint,
                m, n, k, k_start, k_size, vocab.nelems, alpha, beta, index,
                embed, vocab, redux);
    }
}

template<typename T>
void embedding_backward(int starpu_worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, Scalar alpha, Scalar beta,
        const Tile<int64_t> &index, const Tile<T> &embed, const Tile<T> &vocab,
        int redux)
{
    embedding_backward_async<T>(starpu_worker_hint, m, n, k, k_start, k_size,
            alpha, beta, index, embed, vocab, redux);
    nntile::starpu_task_wait_for_all_unless_deferred();
}

#define NNTILE_EMBEDDING_BACKWARD_EXPLICIT(T) \
template void embedding_backward_async<T>(int, Index, Index, Index, Index, \
        Index, Scalar, Scalar, const Tile<int64_t> &, const Tile<T> &, \
        const Tile<T> &, int); \
template void embedding_backward<T>(int, Index, Index, Index, Index, Index, \
        Scalar, Scalar, const Tile<int64_t> &, const Tile<T> &, \
        const Tile<T> &, int);

NNTILE_EMBEDDING_BACKWARD_EXPLICIT(fp32_t)
NNTILE_EMBEDDING_BACKWARD_EXPLICIT(fp32_fast_tf32_t)
NNTILE_EMBEDDING_BACKWARD_EXPLICIT(fp32_fast_fp16_t)
NNTILE_EMBEDDING_BACKWARD_EXPLICIT(fp32_fast_bf16_t)
NNTILE_EMBEDDING_BACKWARD_EXPLICIT(fp64_t)
NNTILE_EMBEDDING_BACKWARD_EXPLICIT(bf16_t)
NNTILE_EMBEDDING_BACKWARD_EXPLICIT(fp16_t)

#undef NNTILE_EMBEDDING_BACKWARD_EXPLICIT

} // namespace nntile::core
