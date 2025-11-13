/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/flash_sdpa_fwd_cudnn.cc
 * Flash attention scaled dot-product attention forward pass using cuDNN
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/config.hh"
#include "nntile/starpu/handle.hh"

namespace nntile::tensor
{

//! Check if tensors match flash_sdpa_fwd_cudnn requirements
static inline void flash_sdpa_fwd_cudnn_check(const TensorTraits &K,
        const TensorTraits &Q, const TensorTraits &mask,
        const TensorTraits &logsumexp, const TensorTraits &V,
        const TensorTraits &A)
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
        throw std::runtime_error("K sequence dimension must match mask axis 0");
    }
    if(Q.shape[1] != mask.shape[1])
    {
        throw std::runtime_error("Q sequence dimension must match mask axis 1");
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

    // Validate basetile compatibility for multi-tile execution
    const auto &K_base = K.basetile_shape;
    const auto &Q_base = Q.basetile_shape;
    const auto &V_base = V.basetile_shape;
    const auto &A_base = A.basetile_shape;
    const auto &mask_base = mask.basetile_shape;
    const auto &logsumexp_base = logsumexp.basetile_shape;

    if(Q_base != A_base)
    {
        throw std::runtime_error("Q and A basetile shapes must match");
    }

    for(Index dim : {Index(0), Index(2), Index(3), Index(4)})
    {
        if(K_base[dim] != Q_base[dim])
        {
            throw std::runtime_error("K and Q basetile shapes mismatch");
        }
        if(V_base[dim] != Q_base[dim])
        {
            throw std::runtime_error("V and Q basetile shapes mismatch");
        }
        if(A_base[dim] != Q_base[dim])
        {
            throw std::runtime_error("A and Q basetile shapes mismatch");
        }
    }

    if(mask_base[0] != K_base[1])
    {
        throw std::runtime_error("mask basetile axis 0 must match K basetile sequence dimension");
    }
    if(mask_base[1] != Q_base[1])
    {
        throw std::runtime_error("mask basetile axis 1 must match Q basetile sequence dimension");
    }

    if(logsumexp_base[0] != Q_base[1]
            || logsumexp_base[1] != Q_base[2]
            || logsumexp_base[2] != Q_base[3]
            || logsumexp_base[3] != Q_base[4])
    {
        throw std::runtime_error("logsumexp basetile shape must match Q basetile shape (excluding head dimension)");
    }

    // Ensure head dimension is not tiled
    if(K_base[0] != K.shape[0] || Q_base[0] != Q.shape[0]
            || V_base[0] != V.shape[0] || A_base[0] != A.shape[0])
    {
        throw std::runtime_error("head dimension must not be tiled for K/Q/V/A");
    }
}

//! Asynchronous tensor-wise flash_sdpa_fwd_cudnn operation
/*! @param[in] K: Key tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] Q: Query tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] mask: Mask tensor [n_seq, n_seq]
 * @param[out] logsumexp: Log-sum-exp statistics [n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] V: Value tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[out] A: Attention output tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * */
template<typename T>
void flash_sdpa_fwd_cudnn_async(const Tensor<T> &K, const Tensor<T> &Q,
        const Tensor<T> &mask, const Tensor<fp32_t> &logsumexp, const Tensor<T> &V,
        const Tensor<T> &A)
{
    // Check inputs (throw exception in case of an error)
    flash_sdpa_fwd_cudnn_check(K, Q, mask, logsumexp, V, A);

    // Get MPI rank and sizes
    int mpi_rank = starpu_mpi_world_rank();

    const Index num_k_seq_tiles = K.grid.shape[1];

    // Loop over output tiles (defines Q and logsumexp tiles)
    for(Index a_linear = 0; a_linear < A.grid.nelems; ++a_linear)
    {
        const auto a_tile_index = A.grid.linear_to_index(a_linear);

        const auto &A_tile_handle = A.get_tile_handle(a_linear);
        const auto &Q_tile_handle = Q.get_tile_handle(a_linear);
        std::vector<Index> logsumexp_tile_index = {
            a_tile_index[1],
            a_tile_index[2],
            a_tile_index[3],
            a_tile_index[4]
        };
        const auto &logsumexp_tile_handle =
                logsumexp.get_tile_handle(logsumexp_tile_index);

        int A_tile_rank = A_tile_handle.mpi_get_rank();

        Q_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        logsumexp_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);

        const auto &Q_traits = Q.get_tile_traits(a_linear);
        const auto &A_traits = A.get_tile_traits(a_linear);
        const auto &logsumexp_traits =
                logsumexp.get_tile_traits(logsumexp_tile_index);

        const Index seq_q = Q_traits.shape[1];
        const Index head = Q_traits.shape[0];
        const Index batch = Q_traits.shape[2] * Q_traits.shape[3] * Q_traits.shape[4];

        // Iterate over all K/V tiles along the sequence dimension
        for(Index k_seq_idx = 0; k_seq_idx < num_k_seq_tiles; ++k_seq_idx)
        {
            auto kv_tile_index = a_tile_index;
            kv_tile_index[1] = k_seq_idx;

            const auto &K_tile_handle = K.get_tile_handle(kv_tile_index);
            const auto &V_tile_handle = V.get_tile_handle(kv_tile_index);
            std::vector<Index> mask_tile_index = {k_seq_idx, a_tile_index[1]};
            const auto &mask_tile_handle = mask.get_tile_handle(mask_tile_index);

            K_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
            V_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
            mask_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);

            if(mpi_rank == A_tile_rank)
            {
                const auto &K_traits = K.get_tile_traits(kv_tile_index);
                const auto &mask_traits = mask.get_tile_traits(mask_tile_index);

                if(K_traits.shape[1] != seq_q)
                {
                    throw std::runtime_error("K tile sequence length mismatches Q tile");
                }
                if(mask_traits.shape[0] != K_traits.shape[1]
                        || mask_traits.shape[1] != seq_q)
                {
                    throw std::runtime_error("Mask tile shape mismatches Q/K tiles");
                }
                if(logsumexp_traits.shape[0] != seq_q)
                {
                    throw std::runtime_error("logsumexp tile shape mismatches Q tile");
                }

                starpu::VariableHandle scratch_logsumexp(
                        sizeof(fp32_t) * logsumexp_traits.nelems);
                starpu::VariableHandle scratch_A(
                        sizeof(T) * A_traits.nelems);

                starpu::flash_sdpa_fwd_cudnn.submit<std::tuple<T>>(
                    seq_q, head, batch, K_tile_handle, Q_tile_handle,
                    mask_tile_handle, logsumexp_tile_handle, V_tile_handle,
                    A_tile_handle, scratch_logsumexp, scratch_A);

                scratch_logsumexp.unregister_submit();
                scratch_A.unregister_submit();
            }
        }

        A_tile_handle.mpi_flush();
        logsumexp_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise flash_sdpa_fwd_cudnn operation
/*! @param[in] K: Key tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] Q: Query tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] mask: Mask tensor [n_seq, n_seq]
 * @param[out] logsumexp: Log-sum-exp statistics [n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[in] V: Value tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * @param[out] A: Attention output tensor [head_size, n_seq, n_batch, kv_group_size, n_head_kv]
 * */
template<typename T>
void flash_sdpa_fwd_cudnn(const Tensor<T> &K, const Tensor<T> &Q,
        const Tensor<T> &mask, const Tensor<fp32_t> &logsumexp, const Tensor<T> &V,
        const Tensor<T> &A)
{
    flash_sdpa_fwd_cudnn_async<T>(K, Q, mask, logsumexp, V, A);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

// Explicit instantiation
template
void flash_sdpa_fwd_cudnn_async<bf16_t>(const Tensor<bf16_t> &K,
        const Tensor<bf16_t> &Q, const Tensor<bf16_t> &mask,
        const Tensor<fp32_t> &logsumexp, const Tensor<bf16_t> &V,
        const Tensor<bf16_t> &A);

template
void flash_sdpa_fwd_cudnn_async<fp16_t>(const Tensor<fp16_t> &K,
        const Tensor<fp16_t> &Q, const Tensor<fp16_t> &mask,
        const Tensor<fp32_t> &logsumexp, const Tensor<fp16_t> &V,
        const Tensor<fp16_t> &A);

template
void flash_sdpa_fwd_cudnn<bf16_t>(const Tensor<bf16_t> &K,
        const Tensor<bf16_t> &Q, const Tensor<bf16_t> &mask,
        const Tensor<fp32_t> &logsumexp, const Tensor<bf16_t> &V,
        const Tensor<bf16_t> &A);

template
void flash_sdpa_fwd_cudnn<fp16_t>(const Tensor<fp16_t> &K,
        const Tensor<fp16_t> &Q, const Tensor<fp16_t> &mask,
        const Tensor<fp32_t> &logsumexp, const Tensor<fp16_t> &V,
        const Tensor<fp16_t> &A);

} // namespace nntile::tensor
