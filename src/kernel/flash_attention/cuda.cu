/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/kernel/flash_attention/cuda.cu
 * Flash attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#include "nntile/kernel/flash_attention/cuda.hh"
#include "nntile/kernel/cuda.hh"
#include <cuda_runtime.h>
#include <limits>

namespace nntile::kernel::flash_attention
{

// CUDA kernel for computing attention scores (Q @ K^T)
template<typename T>
__global__ void compute_scores_kernel(
    Index batch, Index num_heads, Index seq_len, Index head_dim,
    const T *Q, const T *K, typename T::repr_t scale, typename T::repr_t *scores,
    typename T::repr_t *max_scores)
{
    using Y = typename T::repr_t;

    // Each thread handles one query position
    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * num_heads * seq_len) return;

    Index b = idx / (num_heads * seq_len);
    Index temp = idx % (num_heads * seq_len);
    Index h = temp / seq_len;
    Index i = temp % seq_len;

    Index base_offset = (b * num_heads + h) * seq_len * head_dim;
    Index scores_offset = idx * seq_len;

    // Use CUDA-compatible negative infinity
    Y max_score = -INFINITY;

    // Compute scores for this query position
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
        scores[scores_offset + j] = score;
        max_score = max(max_score, score);
    }

    max_scores[idx] = max_score;
}

// CUDA kernel for softmax
template<typename T>
__global__ void softmax_kernel(
    Index batch, Index num_heads, Index seq_len,
    typename T::repr_t *scores, const typename T::repr_t *max_scores)
{
    using Y = typename T::repr_t;

    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * num_heads * seq_len) return;

    Index scores_offset = idx * seq_len;
    Y max_score = max_scores[idx];

    // Compute exp and sum
    Y sum_exp = Y{0.0};
    for(Index j = 0; j < seq_len; ++j)
    {
        Y val = exp(scores[scores_offset + j] - max_score);
        scores[scores_offset + j] = val;
        sum_exp += val;
    }

    // Normalize
    for(Index j = 0; j < seq_len; ++j)
    {
        scores[scores_offset + j] = scores[scores_offset + j] / sum_exp;
    }
}

// CUDA kernel for computing output (scores @ V)
template<typename T>
__global__ void compute_output_kernel(
    Index batch, Index num_heads, Index seq_len, Index head_dim,
    const typename T::repr_t *scores, const T *V, T *O)
{
    using Y = typename T::repr_t;

    Index idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * num_heads * seq_len) return;

    Index b = idx / (num_heads * seq_len);
    Index temp = idx % (num_heads * seq_len);
    Index h = temp / seq_len;
    Index i = temp % seq_len;

    Index base_offset = (b * num_heads + h) * seq_len * head_dim;
    Index scores_offset = idx * seq_len;

    // Compute output for this query position
    for(Index d = 0; d < head_dim; ++d)
    {
        Y output_val = Y{0.0};
        for(Index j = 0; j < seq_len; ++j)
        {
            Index v_idx = base_offset + j * head_dim + d;
            Y v_val = static_cast<Y>(V[v_idx]);
            output_val += scores[scores_offset + j] * v_val;
        }
        Index o_idx = base_offset + i * head_dim + d;
        O[o_idx] = static_cast<T>(output_val);
    }
}

template<typename T>
void cuda(cudaStream_t stream, Index batch, Index num_heads, Index seq_len,
        Index head_dim, const T *Q, const T *K, const T *V, Scalar scale, T *O)
    noexcept
//! Vanilla attention implementation on CUDA
/*!
 * This is a straightforward CUDA implementation of attention that serves as
 * a reference. For production use, cuDNN's optimized SDPA should be used.
 *
 * Input shapes:
 *   Q: [batch, num_heads, seq_len, head_dim]
 *   K: [batch, num_heads, seq_len, head_dim]
 *   V: [batch, num_heads, seq_len, head_dim]
 *   O: [batch, num_heads, seq_len, head_dim]
 *
 * @param[in] stream: CUDA stream
 * @param[in] batch: Batch size
 * @param[in] num_heads: Number of attention heads
 * @param[in] seq_len: Sequence length
 * @param[in] head_dim: Head dimension
 * @param[in] Q: Query tensor
 * @param[in] K: Key tensor
 * @param[in] V: Value tensor
 * @param[in] scale: Scaling factor (typically 1/sqrt(head_dim))
 * @param[out] O: Output tensor
 * */
{
    using Y = typename T::repr_t;

    Index total_queries = batch * num_heads * seq_len;
    Index threads = 256;
    Index blocks = (total_queries + threads - 1) / threads;

    // Allocate temporary storage for scores and max values
    Y *d_scores = nullptr;
    Y *d_max_scores = nullptr;

    size_t scores_size = total_queries * seq_len * sizeof(Y);
    size_t max_size = total_queries * sizeof(Y);

    CUDA_CHECK(cudaMalloc(&d_scores, scores_size), "cudaMalloc d_scores");
    CUDA_CHECK(cudaMalloc(&d_max_scores, max_size), "cudaMalloc d_max_scores");

    // Compute attention scores
    compute_scores_kernel<T><<<blocks, threads, 0, stream>>>(
        batch, num_heads, seq_len, head_dim, Q, K, static_cast<Y>(scale),
        d_scores, d_max_scores);

    // Apply softmax
    softmax_kernel<T><<<blocks, threads, 0, stream>>>(
        batch, num_heads, seq_len, d_scores, d_max_scores);

    // Compute output
    compute_output_kernel<T><<<blocks, threads, 0, stream>>>(
        batch, num_heads, seq_len, head_dim, d_scores, V, O);

    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    // Clean up
    CUDA_CHECK(cudaFree(d_scores), "cudaFree d_scores");
    CUDA_CHECK(cudaFree(d_max_scores), "cudaFree d_max_scores");
}

// Explicit instantiation
template
void cuda<fp32_t>(cudaStream_t stream, Index batch, Index num_heads,
        Index seq_len, Index head_dim, const fp32_t *Q, const fp32_t *K,
        const fp32_t *V, Scalar scale, fp32_t *O)
    noexcept;

template
void cuda<fp64_t>(cudaStream_t stream, Index batch, Index num_heads,
        Index seq_len, Index head_dim, const fp64_t *Q, const fp64_t *K,
        const fp64_t *V, Scalar scale, fp64_t *O)
    noexcept;

template
void cuda<bf16_t>(cudaStream_t stream, Index batch, Index num_heads,
        Index seq_len, Index head_dim, const bf16_t *Q, const bf16_t *K,
        const bf16_t *V, Scalar scale, bf16_t *O)
    noexcept;

template
void cuda<fp16_t>(cudaStream_t stream, Index batch, Index num_heads,
        Index seq_len, Index head_dim, const fp16_t *Q, const fp16_t *K,
        const fp16_t *V, Scalar scale, fp16_t *O)
    noexcept;

} // namespace nntile::kernel::flash_attention
