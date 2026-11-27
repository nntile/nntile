/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/embedding_backward.cc
 * Backward embeddings from vocabulary within Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/embedding_backward.hh"
#include "nntile/nnhaul/ops/embedding_backward.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void embedding_backward_async(int worker_hint, Index m, Index n,
        Index k, Index k_start, Index k_size, Index index_range,
        Scalar alpha, Scalar beta,
        const Tile<int64_t> &index, const Tile<T> &embed, const Tile<T> &vocab,
        int redux)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    nnhaul::embedding_backward.submit<std::tuple<T>>(worker_hint,
            m, n, k, k_start, k_size, index_range, vocab.nelems,
            alpha, beta, index, embed, vocab, redux);
}

template<typename T>
void embedding_backward(int worker_hint, Index m, Index n, Index k,
        Index k_start, Index k_size, Index index_range, Scalar alpha,
        Scalar beta,
        const Tile<int64_t> &index, const Tile<T> &embed, const Tile<T> &vocab,
        int redux)
{
    if(redux != 0)
    {
        throw std::runtime_error("redux != 0 is not supported");
    }
    embedding_backward_async<T>(worker_hint, m, n, k, k_start, k_size,
            index_range, alpha, beta, index, embed, vocab, redux);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

#define NNTILE_EMBEDDING_BACKWARD_EXPLICIT(T) \
template void embedding_backward_async<T>(int, Index, Index, Index, Index, \
        Index, Index, Scalar, Scalar, const Tile<int64_t> &, \
        const Tile<T> &, const Tile<T> &, int); \
template void embedding_backward<T>(int, Index, Index, Index, Index, Index, \
        Index, Scalar, Scalar, const Tile<int64_t> &, const Tile<T> &, \
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
