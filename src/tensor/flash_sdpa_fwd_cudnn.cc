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

namespace nntile::tensor
{

//! Check if tensors match flash_sdpa_fwd_cudnn requirements
static inline void flash_sdpa_fwd_cudnn_check(const TensorTraits &K,
        const TensorTraits &Q, const TensorTraits &mask,
        const TensorTraits &logsumexp, const TensorTraits &V,
        const TensorTraits &A)
{
    // All tensors should be 3D
    if(K.ndim != 3)
    {
        throw std::runtime_error("K.ndim != 3");
    }
    if(Q.ndim != 3)
    {
        throw std::runtime_error("Q.ndim != 3");
    }
    if(V.ndim != 3)
    {
        throw std::runtime_error("V.ndim != 3");
    }
    if(A.ndim != 3)
    {
        throw std::runtime_error("A.ndim != 3");
    }
    if(mask.ndim != 3)
    {
        throw std::runtime_error("mask.ndim != 3");
    }
    if(logsumexp.ndim != 2)
    {
        throw std::runtime_error("logsumexp.ndim != 2");
    }

    // Check batch dimension (first dimension for all tensors)
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
    if(K.shape[0] != mask.shape[0])
    {
        throw std::runtime_error("K.shape[0] != mask.shape[0]");
    }
    if(K.shape[0] != logsumexp.shape[0])
    {
        throw std::runtime_error("K.shape[0] != logsumexp.shape[0]");
    }

    // Check sequence dimension (second dimension for K/Q/V/A, second for mask, first for logsumexp)
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
    if(K.shape[1] != mask.shape[1])
    {
        throw std::runtime_error("K.shape[1] != mask.shape[1]");
    }
    if(K.shape[1] != mask.shape[2])
    {
        throw std::runtime_error("K.shape[1] != mask.shape[2]");
    }
    if(K.shape[1] != logsumexp.shape[1])
    {
        throw std::runtime_error("K.shape[1] != logsumexp.shape[1]");
    }

    // Check head dimension (third dimension for K/Q/V/A)
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

    // Check head dimension matches for mask (should be seq x seq)
    if(mask.shape[1] != mask.shape[2])
    {
        throw std::runtime_error("mask.shape[1] != mask.shape[2]");
    }

    // Check that all tensors have single tile (basetile_shape matches shape)
    for(Index i = 0; i < K.ndim; ++i)
    {
        if(K.basetile_shape[i] != K.shape[i])
        {
            throw std::runtime_error("K.basetile_shape[i] != K.shape[i]");
        }
        if(Q.basetile_shape[i] != Q.shape[i])
        {
            throw std::runtime_error("Q.basetile_shape[i] != Q.shape[i]");
        }
        if(V.basetile_shape[i] != V.shape[i])
        {
            throw std::runtime_error("V.basetile_shape[i] != V.shape[i]");
        }
        if(A.basetile_shape[i] != A.shape[i])
        {
            throw std::runtime_error("A.basetile_shape[i] != A.shape[i]");
        }
    }

    for(Index i = 0; i < mask.ndim; ++i)
    {
        if(mask.basetile_shape[i] != mask.shape[i])
        {
            throw std::runtime_error("mask.basetile_shape[i] != mask.shape[i]");
        }
    }

    for(Index i = 0; i < logsumexp.ndim; ++i)
    {
        if(logsumexp.basetile_shape[i] != logsumexp.shape[i])
        {
            throw std::runtime_error("logsumexp.basetile_shape[i] != logsumexp.shape[i]");
        }
    }
}

//! Asynchronous tensor-wise flash_sdpa_fwd_cudnn operation
/*! @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Mask tensor [batch, seq, seq]
 * @param[inout] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
template<typename T>
void flash_sdpa_fwd_cudnn_async(const Tensor<T> &K, const Tensor<T> &Q,
        const Tensor<T> &mask, const Tensor<T> &logsumexp, const Tensor<T> &V,
        const Tensor<T> &A)
{
    // Check inputs (throw exception in case of an error)
    flash_sdpa_fwd_cudnn_check(K, Q, mask, logsumexp, V, A);

    // Get MPI rank and sizes
    int mpi_rank = starpu_mpi_world_rank();

    // Loop through all tiles in the grid
    for(Index i = 0; i < K.grid.nelems; ++i)
    {
        // Get tile handles for all tensors
        auto K_tile_handle = K.get_tile_handle(i);
        auto Q_tile_handle = Q.get_tile_handle(i);
        auto mask_tile_handle = mask.get_tile_handle(i);
        auto logsumexp_tile_handle = logsumexp.get_tile_handle(i);
        auto V_tile_handle = V.get_tile_handle(i);
        auto A_tile_handle = A.get_tile_handle(i);

        // Get destination rank for A tile
        int A_tile_rank = A_tile_handle.mpi_get_rank();

        // Transfer all input tiles to destination rank
        K_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        Q_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        mask_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        logsumexp_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);
        V_tile_handle.mpi_transfer(A_tile_rank, mpi_rank);

        // Execute on destination node
        if(mpi_rank == A_tile_rank)
        {
            // Get tile traits to extract dimensions
            auto K_traits = K.get_tile_traits(i);

            // Extract dimensions for starpu call
            Index seq = K_traits.shape[1];
            Index head = K_traits.shape[2];
            Index batch = K_traits.shape[0];

            // Submit starpu operation
            starpu::flash_sdpa_fwd_cudnn.submit<std::tuple<T>>(
                seq, head, batch, K_tile_handle, Q_tile_handle, mask_tile_handle,
                logsumexp_tile_handle, V_tile_handle, A_tile_handle);
        }

        // Flush cache for output tile
        A_tile_handle.mpi_flush();
    }
}

//! Blocking version of tensor-wise flash_sdpa_fwd_cudnn operation
/*! @param[in] K: Key tensor [batch, seq, head]
 * @param[in] Q: Query tensor [batch, seq, head]
 * @param[in] mask: Mask tensor [batch, seq, seq]
 * @param[inout] logsumexp: Log-sum-exp statistics [batch, seq]
 * @param[in] V: Value tensor [batch, seq, head]
 * @param[out] A: Attention output tensor [batch, seq, head]
 * */
template<typename T>
void flash_sdpa_fwd_cudnn(const Tensor<T> &K, const Tensor<T> &Q,
        const Tensor<T> &mask, const Tensor<T> &logsumexp, const Tensor<T> &V,
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
        const Tensor<bf16_t> &logsumexp, const Tensor<bf16_t> &V,
        const Tensor<bf16_t> &A);

template
void flash_sdpa_fwd_cudnn_async<fp16_t>(const Tensor<fp16_t> &K,
        const Tensor<fp16_t> &Q, const Tensor<fp16_t> &mask,
        const Tensor<fp16_t> &logsumexp, const Tensor<fp16_t> &V,
        const Tensor<fp16_t> &A);

template
void flash_sdpa_fwd_cudnn<bf16_t>(const Tensor<bf16_t> &K,
        const Tensor<bf16_t> &Q, const Tensor<bf16_t> &mask,
        const Tensor<bf16_t> &logsumexp, const Tensor<bf16_t> &V,
        const Tensor<bf16_t> &A);

template
void flash_sdpa_fwd_cudnn<fp16_t>(const Tensor<fp16_t> &K,
        const Tensor<fp16_t> &Q, const Tensor<fp16_t> &mask,
        const Tensor<fp16_t> &logsumexp, const Tensor<fp16_t> &V,
        const Tensor<fp16_t> &A);

} // namespace nntile::tensor
