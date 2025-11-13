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

namespace nntile::tensor
{

static inline void flash_sdpa_bwd_cudnn_check(
        const TensorTraits &K,
        const TensorTraits &Q,
        const TensorTraits &V,
        const TensorTraits &O,
        const TensorTraits &dO,
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
    check_5d(O, "O");
    check_5d(dO, "dO");
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
    check_equal_shape(K, O, "K", "O");
    check_equal_shape(K, dO, "K", "dO");
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

static inline void ensure_single_tile(const TensorTraits &traits,
        const char *name)
{
    if (traits.grid.nelems != 1) {
        throw std::runtime_error(std::string(name) +
            " must consist of a single tile for flash SDPA backward");
    }
}

template<typename T>
void flash_sdpa_bwd_cudnn_async(
    const Tensor<T> &K,
    const Tensor<T> &Q,
    const Tensor<T> &V,
    const Tensor<T> &O,
    const Tensor<T> &dO,
    const Tensor<T> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<T> &dK,
    const Tensor<T> &dQ,
    const Tensor<T> &dV)
{
    flash_sdpa_bwd_cudnn_check(K, Q, V, O, dO, mask, logsumexp, dK, dQ, dV);

    ensure_single_tile(K, "K");
    ensure_single_tile(Q, "Q");
    ensure_single_tile(V, "V");
    ensure_single_tile(O, "O");
    ensure_single_tile(dO, "dO");
    ensure_single_tile(dK, "dK");
    ensure_single_tile(dQ, "dQ");
    ensure_single_tile(dV, "dV");
    ensure_single_tile(mask, "mask");
    ensure_single_tile(logsumexp, "logsumexp");

    const auto &K_handle = K.get_tile_handle(0);
    const auto &Q_handle = Q.get_tile_handle(0);
    const auto &V_handle = V.get_tile_handle(0);
    const auto &O_handle = O.get_tile_handle(0);
    const auto &dO_handle = dO.get_tile_handle(0);
    const auto &mask_handle = mask.get_tile_handle(0);
    const auto &logsumexp_handle = logsumexp.get_tile_handle(0);
    const auto &dK_handle = dK.get_tile_handle(0);
    const auto &dQ_handle = dQ.get_tile_handle(0);
    const auto &dV_handle = dV.get_tile_handle(0);

    Index seq = K.shape[1];
    Index head = K.shape[0];
    Index batch = K.shape[2] * K.shape[3] * K.shape[4];

    starpu::flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch,
        K_handle, Q_handle, V_handle, O_handle, dO_handle,
        mask_handle, logsumexp_handle,
        dK_handle, dQ_handle, dV_handle
    );
}

template<typename T>
void flash_sdpa_bwd_cudnn(
    const Tensor<T> &K,
    const Tensor<T> &Q,
    const Tensor<T> &V,
    const Tensor<T> &O,
    const Tensor<T> &dO,
    const Tensor<T> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<T> &dK,
    const Tensor<T> &dQ,
    const Tensor<T> &dV)
{
    flash_sdpa_bwd_cudnn_async<T>(
        K, Q, V, O, dO, mask, logsumexp, dK, dQ, dV);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
}

template
void flash_sdpa_bwd_cudnn_async<bf16_t>(
    const Tensor<bf16_t> &K,
    const Tensor<bf16_t> &Q,
    const Tensor<bf16_t> &V,
    const Tensor<bf16_t> &O,
    const Tensor<bf16_t> &dO,
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
    const Tensor<fp16_t> &O,
    const Tensor<fp16_t> &dO,
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
    const Tensor<bf16_t> &O,
    const Tensor<bf16_t> &dO,
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
    const Tensor<fp16_t> &O,
    const Tensor<fp16_t> &dO,
    const Tensor<fp16_t> &mask,
    const Tensor<fp32_t> &logsumexp,
    const Tensor<fp16_t> &dK,
    const Tensor<fp16_t> &dQ,
    const Tensor<fp16_t> &dV);

} // namespace nntile::tensor
