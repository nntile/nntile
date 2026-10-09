/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/embedding_backward/cpu.cc
 * Backward of embeddings from vocabulary within buffers
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/embedding_backward/cpu.hh"
#include "nntile/kernel/cpu.hh"
#include <cstring>

namespace nntile::kernel::embedding_backward
{

template<typename T>
void cpu(Index m, Index n, Index k, Index k_start, Index k_size,
        Index vocab_nelems, Scalar alpha, Scalar beta, const int64_t *index_,
        const T *embed, T *vocab)
    noexcept
//! Accumulate gradients of embeddings into vocabulary
/*! Does the following operation:
 *      vocab = beta * vocab + alpha * scatter(embed)
 * with beta in {0,1}. If beta=0, the entire vocab buffer is zeroed first.
 * */
{
    using Y = typename T::repr_t;
    using I = typename CPUComputeType<int64_t>::value;
    auto index = reinterpret_cast<const I *>(index_);
    const Y alpha_ = static_cast<Y>(alpha);
    if(beta == Scalar{0.0})
    {
        std::memset(vocab, 0, static_cast<size_t>(vocab_nelems) * sizeof(T));
    }
    for(Index i2 = 0; i2 < n; ++i2)
    {
        for(Index i1 = 0; i1 < m; ++i1)
        {
            T *vocab_slice = vocab + k_size*index[i2*m+i1];
            const T *embed_slice = embed + (i2*k+k_start)*m + i1;
            for(Index i0 = 0; i0 < k_size; ++i0)
            {
                vocab_slice[i0] = static_cast<T>(
                    Y{vocab_slice[i0]} + alpha_ * Y{embed_slice[i0*m]});
            }
        }
    }
}

template
void cpu<fp32_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Index vocab_nelems, Scalar alpha, Scalar beta, const int64_t *index,
        const fp32_t *embed, fp32_t *vocab)
    noexcept;

template
void cpu<fp64_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Index vocab_nelems, Scalar alpha, Scalar beta, const int64_t *index,
        const fp64_t *embed, fp64_t *vocab)
    noexcept;

template
void cpu<bf16_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Index vocab_nelems, Scalar alpha, Scalar beta, const int64_t *index,
        const bf16_t *embed, bf16_t *vocab)
    noexcept;

template
void cpu<fp16_t>(Index m, Index n, Index k, Index k_start, Index k_size,
        Index vocab_nelems, Scalar alpha, Scalar beta, const int64_t *index,
        const fp16_t *embed, fp16_t *vocab)
    noexcept;

} // namespace nntile::kernel::embedding_backward
