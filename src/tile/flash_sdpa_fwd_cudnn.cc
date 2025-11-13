/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/flash_sdpa_fwd_cudnn.cc
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#include "nntile/tile/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/handle.hh"
#include "nntile/constants.hh"

namespace nntile::tile
{

//! Check if tensors match flash_sdpa_fwd_cudnn requirements
static inline void flash_sdpa_fwd_cudnn_check(const TileTraits &K,
        const TileTraits &Q, const TileTraits &mask,
        const TileTraits &logsumexp, const TileTraits &V,
        const TileTraits &A)
{
    // All tensors should be 5D for K/Q/V/A, 2D for mask, 4D for logsumexp
    if(K.ndim != 5)
    {
        throw std::runtime_error("K.ndim != 5");
    }
    if(Q.ndim != 5)
    {
        throw std::runtime_error("Q.ndim != 5");
    }
    if(V.ndim != 5)
    {
        throw std::runtime_error("V.ndim != 5");
    }
    if(A.ndim != 5)
    {
        throw std::runtime_error("A.ndim != 5");
    }
    if(mask.ndim != 2)
    {
        throw std::runtime_error("mask.ndim != 2");
    }
    if(logsumexp.ndim != 4)
    {
        throw std::runtime_error("logsumexp.ndim != 4");
    }

    // Check head_size dimension (first dimension for K/Q/V/A)
    if(K.shape[0] != Q.shape[0])
    {
        throw std::runtime_error("K.shape[0] != Q.shape[0]");
    }
    if(K.shape[0] != V.shape[0])
    {
        throw std::runtime_error("K.shape[0] != V.shape[0]");
    }
    if(K.shape[0] != A.shape[0])
    {
        throw std::runtime_error("K.shape[0] != A.shape[0]");
    }

    // Check sequence dimension (second dimension for K/Q/V/A, both dims for mask, first for logsumexp)
    if(K.shape[1] != Q.shape[1])
    {
        throw std::runtime_error("K.shape[1] != Q.shape[1]");
    }
    if(K.shape[1] != V.shape[1])
    {
        throw std::runtime_error("K.shape[1] != V.shape[1]");
    }
    if(K.shape[1] != A.shape[1])
    {
        throw std::runtime_error("K.shape[1] != A.shape[1]");
    }
    if(K.shape[1] != mask.shape[0])
    {
        throw std::runtime_error("K.shape[1] != mask.shape[0]");
    }
    if(K.shape[1] != mask.shape[1])
    {
        throw std::runtime_error("K.shape[1] != mask.shape[1]");
    }
    if(K.shape[1] != logsumexp.shape[0])
    {
        throw std::runtime_error("K.shape[1] != logsumexp.shape[0]");
    }

    // Check batch dimension (third dimension for K/Q/V/A, second for logsumexp)
    if(K.shape[2] != Q.shape[2])
    {
        throw std::runtime_error("K.shape[2] != Q.shape[2]");
    }
    if(K.shape[2] != V.shape[2])
    {
        throw std::runtime_error("K.shape[2] != V.shape[2]");
    }
    if(K.shape[2] != A.shape[2])
    {
        throw std::runtime_error("K.shape[2] != A.shape[2]");
    }
    if(K.shape[2] != logsumexp.shape[1])
    {
        throw std::runtime_error("K.shape[2] != logsumexp.shape[1]");
    }

    // Check kv_group_size dimension (fourth dimension for K/Q/V/A, third for logsumexp)
    if(K.shape[3] != Q.shape[3])
    {
        throw std::runtime_error("K.shape[3] != Q.shape[3]");
    }
    if(K.shape[3] != V.shape[3])
    {
        throw std::runtime_error("K.shape[3] != V.shape[3]");
    }
    if(K.shape[3] != A.shape[3])
    {
        throw std::runtime_error("K.shape[3] != A.shape[3]");
    }
    if(K.shape[3] != logsumexp.shape[2])
    {
        throw std::runtime_error("K.shape[3] != logsumexp.shape[2]");
    }

    // Check n_head_kv dimension (fifth dimension for K/Q/V/A, fourth for logsumexp)
    if(K.shape[4] != Q.shape[4])
    {
        throw std::runtime_error("K.shape[4] != Q.shape[4]");
    }
    if(K.shape[4] != V.shape[4])
    {
        throw std::runtime_error("K.shape[4] != V.shape[4]");
    }
    if(K.shape[4] != A.shape[4])
    {
        throw std::runtime_error("K.shape[4] != A.shape[4]");
    }
    if(K.shape[4] != logsumexp.shape[3])
    {
        throw std::runtime_error("K.shape[4] != logsumexp.shape[3]");
    }

    // Mask dimensions are already checked above (both should equal seq)
}

//! Asynchronous tile-wise flash_sdpa_fwd_cudnn operation
/*! @param[in] K: Key tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] Q: Query tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] mask: Mask tensor [n_seq, n_seq]
 * @param[inout] logsumexp: Log-sum-exp statistics [n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] V: Value tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[out] A: Attention output tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * */
template<typename T>
void flash_sdpa_fwd_cudnn_async(const Tile<T> &K, const Tile<T> &Q,
        const Tile<T> &mask, const Tile<fp32_t> &logsumexp, const Tile<T> &V,
        const Tile<T> &A)
{
    // Check inputs (throw exception in case of an error)
    flash_sdpa_fwd_cudnn_check(K, Q, mask, logsumexp, V, A);

    // Extract dimensions for starpu call
    Index seq = K.shape[1];
    Index head = K.shape[0];
    // Combine batch dimensions: n_batch * kv_group_size * n_head_kv -> batch
    Index batch = K.shape[2] * K.shape[3] * K.shape[4];

    starpu::VariableHandle scratch_logsumexp(sizeof(fp32_t) * logsumexp.nelems);
    starpu::VariableHandle scratch_A(sizeof(T) * A.nelems);

    // Insert task
    starpu::flash_sdpa_fwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch, K, Q, mask, logsumexp, V, A,
        scratch_logsumexp, scratch_A);

    scratch_logsumexp.unregister_submit();
    scratch_A.unregister_submit();
}

//! Blocking version of tile-wise flash_sdpa_fwd_cudnn operation
/*! @param[in] K: Key tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] Q: Query tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] mask: Mask tensor [n_seq, n_seq]
 * @param[inout] logsumexp: Log-sum-exp statistics [n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] V: Value tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[out] A: Attention output tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * */
template<typename T>
void flash_sdpa_fwd_cudnn(const Tile<T> &K, const Tile<T> &Q,
        const Tile<T> &mask, const Tile<fp32_t> &logsumexp, const Tile<T> &V,
        const Tile<T> &A)
{
    flash_sdpa_fwd_cudnn_async<T>(K, Q, mask, logsumexp, V, A);
    starpu_task_wait_for_all();
}

// Explicit instantiation
template
void flash_sdpa_fwd_cudnn_async<bf16_t>(const Tile<bf16_t> &K,
        const Tile<bf16_t> &Q, const Tile<bf16_t> &mask,
        const Tile<fp32_t> &logsumexp, const Tile<bf16_t> &V,
        const Tile<bf16_t> &A);

template
void flash_sdpa_fwd_cudnn_async<fp16_t>(const Tile<fp16_t> &K,
        const Tile<fp16_t> &Q, const Tile<fp16_t> &mask,
        const Tile<fp32_t> &logsumexp, const Tile<fp16_t> &V,
        const Tile<fp16_t> &A);

template
void flash_sdpa_fwd_cudnn<bf16_t>(const Tile<bf16_t> &K,
        const Tile<bf16_t> &Q, const Tile<bf16_t> &mask,
        const Tile<fp32_t> &logsumexp, const Tile<bf16_t> &V,
        const Tile<bf16_t> &A);

template
void flash_sdpa_fwd_cudnn<fp16_t>(const Tile<fp16_t> &K,
        const Tile<fp16_t> &Q, const Tile<fp16_t> &mask,
        const Tile<fp32_t> &logsumexp, const Tile<fp16_t> &V,
        const Tile<fp16_t> &A);

} // namespace nntile::tile
