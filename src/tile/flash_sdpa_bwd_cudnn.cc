/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/tile/flash_sdpa_bwd_cudnn.cc
 * Flash attention scaled dot-product attention backward pass using cuDNN
 *
 * @version 1.1.0
 * */

#include "nntile/tile/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/config.hh"
#include "nntile/starpu/handle.hh"

namespace nntile::tile
{

static inline void flash_sdpa_bwd_cudnn_check(
        const TileTraits &K,
        const TileTraits &Q,
        const TileTraits &V,
        const TileTraits &A,
        const TileTraits &dA,
        const TileTraits &mask,
        const TileTraits &logsumexp,
        const TileTraits &dK,
        const TileTraits &dQ,
        const TileTraits &dV)
{
    auto check_5d = [](const TileTraits &t, const char *name) {
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

    auto check_equal_shape = [](const TileTraits &a, const TileTraits &b,
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
}

template<typename T>
void flash_sdpa_bwd_cudnn_async(
    const Tile<T> &K,
    const Tile<T> &Q,
    const Tile<T> &V,
    const Tile<T> &A,
    const Tile<T> &dA,
    const Tile<T> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<T> &dK,
    const Tile<T> &dQ,
    const Tile<T> &dV)
{
    flash_sdpa_bwd_cudnn_check(K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);

    Index seq = K.shape[1];
    Index head = K.shape[0];
    Index batch = K.shape[2] * K.shape[3] * K.shape[4];
    int mpi_rank = starpu_mpi_world_rank();
    int dq_rank = dQ.mpi_get_rank();
    K.mpi_transfer(dq_rank, mpi_rank);
    Q.mpi_transfer(dq_rank, mpi_rank);
    V.mpi_transfer(dq_rank, mpi_rank);
    A.mpi_transfer(dq_rank, mpi_rank);
    dA.mpi_transfer(dq_rank, mpi_rank);
    mask.mpi_transfer(dq_rank, mpi_rank);
    logsumexp.mpi_transfer(dq_rank, mpi_rank);
    dK.mpi_transfer(dq_rank, mpi_rank);
    dV.mpi_transfer(dq_rank, mpi_rank);
    if(mpi_rank != dq_rank)
    {
        return;
    }

    starpu::VariableHandle scratch_dK(sizeof(T) * dK.nelems);
    starpu::VariableHandle scratch_dQ(sizeof(T) * dQ.nelems);
    starpu::VariableHandle scratch_dV(sizeof(T) * dV.nelems);

    starpu::flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch,
        K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV,
        scratch_dK, scratch_dQ, scratch_dV);

    scratch_dK.unregister_submit();
    scratch_dQ.unregister_submit();
    scratch_dV.unregister_submit();
}

template<typename T>
void flash_sdpa_bwd_cudnn(
    const Tile<T> &K,
    const Tile<T> &Q,
    const Tile<T> &V,
    const Tile<T> &A,
    const Tile<T> &dA,
    const Tile<T> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<T> &dK,
    const Tile<T> &dQ,
    const Tile<T> &dV)
{
    flash_sdpa_bwd_cudnn_async<T>(
        K, Q, V, A, dA, mask, logsumexp, dK, dQ, dV);
    starpu_task_wait_for_all();
}

template
void flash_sdpa_bwd_cudnn_async<bf16_t>(
    const Tile<bf16_t> &K,
    const Tile<bf16_t> &Q,
    const Tile<bf16_t> &V,
    const Tile<bf16_t> &A,
    const Tile<bf16_t> &dA,
    const Tile<bf16_t> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<bf16_t> &dK,
    const Tile<bf16_t> &dQ,
    const Tile<bf16_t> &dV);

template
void flash_sdpa_bwd_cudnn_async<fp16_t>(
    const Tile<fp16_t> &K,
    const Tile<fp16_t> &Q,
    const Tile<fp16_t> &V,
    const Tile<fp16_t> &A,
    const Tile<fp16_t> &dA,
    const Tile<fp16_t> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<fp16_t> &dK,
    const Tile<fp16_t> &dQ,
    const Tile<fp16_t> &dV);

template
void flash_sdpa_bwd_cudnn<bf16_t>(
    const Tile<bf16_t> &K,
    const Tile<bf16_t> &Q,
    const Tile<bf16_t> &V,
    const Tile<bf16_t> &A,
    const Tile<bf16_t> &dA,
    const Tile<bf16_t> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<bf16_t> &dK,
    const Tile<bf16_t> &dQ,
    const Tile<bf16_t> &dV);

template
void flash_sdpa_bwd_cudnn<fp16_t>(
    const Tile<fp16_t> &K,
    const Tile<fp16_t> &Q,
    const Tile<fp16_t> &V,
    const Tile<fp16_t> &A,
    const Tile<fp16_t> &dA,
    const Tile<fp16_t> &mask,
    const Tile<fp32_t> &logsumexp,
    const Tile<fp16_t> &dK,
    const Tile<fp16_t> &dQ,
    const Tile<fp16_t> &dV);

} // namespace nntile::tile
