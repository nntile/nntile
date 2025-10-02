/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_attention/cpu.cc
 * Flash attention forward pass (vanilla attention implementation as reference)
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/flash_attention/cpu.hh"
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include "nntile/kernel/cpu.hh"

namespace nntile::kernel::flash_attention
{

template<typename T>
void cpu(Index batch, Index num_heads, Index seq_len, Index head_dim,
        const T *Q, const T *K, const T *V, Scalar scale_, T *O)
    noexcept
//! Vanilla attention implementation as reference for flash attention
/*!
 * Computes attention: O = softmax(Q @ K^T / scale) @ V
 * 
 * Input shapes:
 *   Q: [batch, num_heads, seq_len, head_dim]
 *   K: [batch, num_heads, seq_len, head_dim]
 *   V: [batch, num_heads, seq_len, head_dim]
 *   O: [batch, num_heads, seq_len, head_dim]
 *
 * @param[in] batch: Batch size
 * @param[in] num_heads: Number of attention heads
 * @param[in] seq_len: Sequence length
 * @param[in] head_dim: Head dimension
 * @param[in] Q: Query tensor
 * @param[in] K: Key tensor
 * @param[in] V: Value tensor
 * @param[in] scale_: Scaling factor (typically 1/sqrt(head_dim))
 * @param[out] O: Output tensor
 * */
{
    using Y = typename T::repr_t;
    const Y scale{scale_};
    
    // Iterate over batch and heads
    for(Index b = 0; b < batch; ++b)
    {
        for(Index h = 0; h < num_heads; ++h)
        {
            // Base offset for current batch and head
            Index base_offset = (b * num_heads + h) * seq_len * head_dim;
            
            // For each query position
            for(Index i = 0; i < seq_len; ++i)
            {
                // Compute attention scores: Q[i] @ K^T
                // scores[j] = Q[i] · K[j] / scale
                std::vector<Y> scores(seq_len);
                Y max_score = -std::numeric_limits<Y>::infinity();
                
                for(Index j = 0; j < seq_len; ++j)
                {
                    Y score = Y{0.0};
                    // Dot product between Q[i] and K[j]
                    for(Index d = 0; d < head_dim; ++d)
                    {
                        Index q_idx = base_offset + i * head_dim + d;
                        Index k_idx = base_offset + j * head_dim + d;
                        Y q_val = static_cast<Y>(Q[q_idx]);
                        Y k_val = static_cast<Y>(K[k_idx]);
                        score += q_val * k_val;
                    }
                    score = score * scale;
                    scores[j] = score;
                    max_score = std::max(max_score, score);
                }
                
                // Apply softmax: compute exp and sum
                Y sum_exp = Y{0.0};
                for(Index j = 0; j < seq_len; ++j)
                {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                
                // Normalize
                for(Index j = 0; j < seq_len; ++j)
                {
                    scores[j] = scores[j] / sum_exp;
                }
                
                // Compute output: O[i] = sum_j(scores[j] * V[j])
                for(Index d = 0; d < head_dim; ++d)
                {
                    Y output_val = Y{0.0};
                    for(Index j = 0; j < seq_len; ++j)
                    {
                        Index v_idx = base_offset + j * head_dim + d;
                        Y v_val = static_cast<Y>(V[v_idx]);
                        output_val += scores[j] * v_val;
                    }
                    Index o_idx = base_offset + i * head_dim + d;
                    O[o_idx] = static_cast<T>(output_val);
                }
            }
        }
    }
}

// Explicit instantiation
template
void cpu<fp32_t>(Index batch, Index num_heads, Index seq_len, Index head_dim,
        const fp32_t *Q, const fp32_t *K, const fp32_t *V, Scalar scale,
        fp32_t *O)
    noexcept;

template
void cpu<fp64_t>(Index batch, Index num_heads, Index seq_len, Index head_dim,
        const fp64_t *Q, const fp64_t *K, const fp64_t *V, Scalar scale,
        fp64_t *O)
    noexcept;

template
void cpu<bf16_t>(Index batch, Index num_heads, Index seq_len, Index head_dim,
        const bf16_t *Q, const bf16_t *K, const bf16_t *V, Scalar scale,
        bf16_t *O)
    noexcept;

template
void cpu<fp16_t>(Index batch, Index num_heads, Index seq_len, Index head_dim,
        const fp16_t *Q, const fp16_t *K, const fp16_t *V, Scalar scale,
        fp16_t *O)
    noexcept;

} // namespace nntile::kernel::flash_attention
