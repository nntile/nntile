/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tensor/flash_sdpa_bwd_cudnn.cc
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/config.hh"
#include "nntile/starpu/handle.hh"

namespace nntile::tensor
{

static inline void flash_sdpa_bwd_cudnn_check(
        const TensorTraits &K,
        const TensorTraits &Q,
        const TensorTraits &V,
        const TensorTraits &A,
        const TensorTraits &dA,
        const TensorTraits &mask,
        const TensorTraits &logsumexp,
        const TensorTraits &dK,
        const TensorTraits &dQ,
        const TensorTraits &dV)
{
    auto check_5d = [](const TensorTraits &t, const char *name) {
        if (t.ndim != 5) {
            throw std::runtime_error(std::string(name) + ".ndim != 5");
        }
    };
    check_5d(K, "K");
    check_5d(Q, "Q");
    check_5d(V, "V");
    check_5d(A, "A");
    check_5d(dA, "dA");
    check_5d(dK, "dK");
    check_5d(dQ, "dQ");
    check_5d(dV, "dV");

    if (mask.ndim != 2) {
        throw std::runtime_error("mask.ndim != 2");
    }
    if (logsumexp.ndim != 4) {
        throw std::runtime_error("logsumexp.ndim != 4");
    }

    auto check_equal_shape = [](const TensorTraits &a, const TensorTraits &b,
                                const char *lhs, const char *rhs) {
        if (a.shape != b.shape) {
            throw std::runtime_error(std::string(lhs) + ".shape != "
                    + rhs + ".shape");
        }
    };

    check_equal_shape(K, Q, "K", "Q");
    check_equal_shape(K, V, "K", "V");
    check_equal_shape(K, A, "K", "A");
    check_equal_shape(K, dA, "K", "dA");
    check_equal_shape(K, dK, "K", "dK");
    check_equal_shape(K, dQ, "K", "dQ");
    check_equal_shape(K, dV, "K", "dV");

    if (K.shape[1] != mask.shape[0] || K.shape[1] != mask.shape[1]) {
        throw std::runtime_error("Mask shape mismatch");
    }

    if (K.shape[1] != logsumexp.shape[0]) {
        throw std::runtime_error("logsumexp axis 0 must match sequence length");
    }
    if (K.shape[2] != logsumexp.shape[1]) {
        throw std::runtime_error("logsumexp axis 1 must match batch");
    }
    if (K.shape[3] != logsumexp.shape[2]) {
        throw std::runtime_error("logsumexp axis 2 must match kv_group_size");
    }
    if (K.shape[4] != logsumexp.shape[3]) {
        throw std::runtime_error("logsumexp axis 3 must match n_head_kv");
    }

    // Validate basetile compatibility for multi-tile execution
    const auto &K_base = K.basetile_shape;
    const auto &Q_base = Q.basetile_shape;
    const auto &V_base = V.basetile_shape;
    const auto &A_base = A.basetile_shape;
    const auto &dA_base = dA.basetile_shape;
    const auto &dK_base = dK.basetile_shape;
    const auto &dQ_base = dQ.basetile_shape;
    const auto &dV_base = dV.basetile_shape;
    const auto &mask_base = mask.basetile_shape;
    const auto &logsumexp_base = logsumexp.basetile_shape;

    if(Q_base != dQ_base || Q_base != A_base || Q_base != dA_base)
    {
        throw std::runtime_error("Q, dQ, A, dA basetile shapes must match");
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
        if(dK_base[dim] != Q_base[dim])
        {
            throw std::runtime_error("dK and Q basetile shapes mismatch");
        }
        if(dV_base[dim] != Q_base[dim])
        {
            throw std::runtime_error("dV and Q basetile shapes mismatch");
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
            || V_base[0] != V.shape[0] || A_base[0] != A.shape[0]
            || dA_base[0] != dA.shape[0] || dK_base[0] != dK.shape[0]
            || dQ_base[0] != dQ.shape[0] || dV_base[0] != dV.shape[0])
    {
        throw std::runtime_error("head dimension must not be tiled");
    }
}


template<typename T>
void flash_sdpa_bwd_cudnn_async(
    const Tensor<T> &K,
    const Tensor<T> &Q,
    const Tensor<T> &V,
    const Tensor<T> &A,
    const Tensor<T> &dA,
    const Tensor<T> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<T> &dK,
    const Tensor<T> &dQ,
    const Tensor<T> &dV)
{
    // Check inputs (throw exception in case of an error)
    flash_sdpa_bwd_cudnn_check(K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);

    // Get MPI rank and sizes
    int mpi_rank = starpu_mpi_world_rank();

    const Index num_k_seq_tiles = K.grid.shape[1];

    // Loop over output tiles (defines dQ and related tiles)
    for(Index dq_linear = 0; dq_linear < dQ.grid.nelems; ++dq_linear)
    {
        const auto dq_tile_index = dQ.grid.linear_to_index(dq_linear);

        const auto &dQ_handle = dQ.get_tile_handle(dq_linear);
        const auto &Q_handle = Q.get_tile_handle(dq_linear);
        const auto &A_handle = A.get_tile_handle(dq_linear);
        const auto &dA_handle = dA.get_tile_handle(dq_linear);
        std::vector<Index> logsumexp_tile_index = {
            dq_tile_index[1],
            dq_tile_index[2],
            dq_tile_index[3],
            dq_tile_index[4]
        };
        const auto &logsumexp_handle =
                logsumexp.get_tile_handle(logsumexp_tile_index);

        int dQ_tile_rank = dQ_handle.mpi_get_rank();

        Q_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
        A_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
        dA_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
        logsumexp_handle.mpi_transfer(dQ_tile_rank, mpi_rank);

        const auto &dQ_traits = dQ.get_tile_traits(dq_linear);
        const auto &Q_traits = Q.get_tile_traits(dq_linear);
        const auto &A_traits = A.get_tile_traits(dq_linear);
        const auto &dA_traits = dA.get_tile_traits(dq_linear);
        const auto &logsumexp_traits =
                logsumexp.get_tile_traits(logsumexp_tile_index);

        const Index seq = dQ_traits.shape[1];
        const Index head = dQ_traits.shape[0];
        const Index batch = dQ_traits.shape[2] * dQ_traits.shape[3] * dQ_traits.shape[4];

        // Iterate over all K/V tiles along the sequence dimension
        for(Index k_seq_idx = 0; k_seq_idx < num_k_seq_tiles; ++k_seq_idx)
        {
            auto kv_tile_index = dq_tile_index;
            kv_tile_index[1] = k_seq_idx;

            const auto &K_handle = K.get_tile_handle(kv_tile_index);
            const auto &V_handle = V.get_tile_handle(kv_tile_index);
            const auto &dK_handle = dK.get_tile_handle(kv_tile_index);
            const auto &dV_handle = dV.get_tile_handle(kv_tile_index);
            std::vector<Index> mask_tile_index = {k_seq_idx, dq_tile_index[1]};
            const auto &mask_handle = mask.get_tile_handle(mask_tile_index);

            K_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
            V_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
            dK_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
            dV_handle.mpi_transfer(dQ_tile_rank, mpi_rank);
            mask_handle.mpi_transfer(dQ_tile_rank, mpi_rank);

            if(mpi_rank == dQ_tile_rank)
            {
                const auto &K_traits = K.get_tile_traits(kv_tile_index);
                const auto &V_traits = V.get_tile_traits(kv_tile_index);
                const auto &dK_traits = dK.get_tile_traits(kv_tile_index);
                const auto &dV_traits = dV.get_tile_traits(kv_tile_index);
                const auto &mask_traits = mask.get_tile_traits(mask_tile_index);

                if(K_traits.shape[1] != seq)
                {
                    throw std::runtime_error("K tile sequence length mismatches dQ tile");
                }
                if(mask_traits.shape[0] != K_traits.shape[1]
                        || mask_traits.shape[1] != seq)
                {
                    throw std::runtime_error("Mask tile shape mismatches dQ/K tiles");
                }
                if(logsumexp_traits.shape[0] != seq)
                {
                    throw std::runtime_error("logsumexp tile shape mismatches dQ tile");
                }

                starpu::VariableHandle scratch_dK(sizeof(T) * dK_traits.nelems);
                starpu::VariableHandle scratch_dQ(sizeof(T) * dQ_traits.nelems);
                starpu::VariableHandle scratch_dV(sizeof(T) * dV_traits.nelems);

                starpu::flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
                    seq, head, batch,
                    K_handle, Q_handle, V_handle, A_handle, dA_handle,
                    mask_handle, logsumexp_handle,
                    dK_handle, dQ_handle, dV_handle,
                    scratch_dK, scratch_dQ, scratch_dV
                );

                scratch_dK.unregister_submit();
                scratch_dQ.unregister_submit();
                scratch_dV.unregister_submit();
            }
        }

        dQ_handle.mpi_flush();
        logsumexp_handle.mpi_flush();
    }
}

template<typename T>
void flash_sdpa_bwd_cudnn(
    const Tensor<T> &K,
    const Tensor<T> &Q,
    const Tensor<T> &V,
    const Tensor<T> &A,
    const Tensor<T> &dA,
    const Tensor<T> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<T> &dK,
    const Tensor<T> &dQ,
    const Tensor<T> &dV)
{
    flash_sdpa_bwd_cudnn_async<T>(
        K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

template
void flash_sdpa_bwd_cudnn_async<bf16_t>(
    const Tensor<bf16_t> &K,
    const Tensor<bf16_t> &Q,
    const Tensor<bf16_t> &V,
    const Tensor<bf16_t> &A,
    const Tensor<bf16_t> &dA,
    const Tensor<bf16_t> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<bf16_t> &dK,
    const Tensor<bf16_t> &dQ,
    const Tensor<bf16_t> &dV);

template
void flash_sdpa_bwd_cudnn_async<fp16_t>(
    const Tensor<fp16_t> &K,
    const Tensor<fp16_t> &Q,
    const Tensor<fp16_t> &V,
    const Tensor<fp16_t> &A,
    const Tensor<fp16_t> &dA,
    const Tensor<fp16_t> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<fp16_t> &dK,
    const Tensor<fp16_t> &dQ,
    const Tensor<fp16_t> &dV);

template
void flash_sdpa_bwd_cudnn<bf16_t>(
    const Tensor<bf16_t> &K,
    const Tensor<bf16_t> &Q,
    const Tensor<bf16_t> &V,
    const Tensor<bf16_t> &A,
    const Tensor<bf16_t> &dA,
    const Tensor<bf16_t> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<bf16_t> &dK,
    const Tensor<bf16_t> &dQ,
    const Tensor<bf16_t> &dV);

template
void flash_sdpa_bwd_cudnn<fp16_t>(
    const Tensor<fp16_t> &K,
    const Tensor<fp16_t> &Q,
    const Tensor<fp16_t> &V,
    const Tensor<fp16_t> &A,
    const Tensor<fp16_t> &dA,
    const Tensor<fp16_t> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<fp16_t> &dK,
    const Tensor<fp16_t> &dQ,
    const Tensor<fp16_t> &dV);

} // namespace nntile::tensor
