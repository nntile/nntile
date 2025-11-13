/*! @file tests/tensor/flash_sdpa_bwd_cudnn.cc
 * Flash attention SDPA backward pass on Tensor<T>
 */

#include "nntile/context.hh"
#include "nntile/tensor/flash_sdpa_bwd_cudnn.hh"
#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void copy_vector_to_tensor(Tensor<T> &tensor,
        const std::vector<typename T::repr_t> &values)
{
    auto tile = tensor.get_tile(0);
    auto local = tile.acquire(STARPU_W);
    for(Index i = 0; i < tile.nelems; ++i)
    {
        local[i] = T(values[i]);
    }
    local.release();
}

template<typename T>
std::vector<typename T::repr_t> tensor_to_vector(const Tensor<T> &tensor)
{
    auto tile = tensor.get_tile(0);
    auto local = tile.acquire(STARPU_R);
    std::vector<typename T::repr_t> out(tile.nelems);
    for(Index i = 0; i < tile.nelems; ++i)
    {
        out[i] = static_cast<typename T::repr_t>(local[i]);
    }
    local.release();
    return out;
}

template<typename T>
void check_single_tile()
{
    using Y = typename T::repr_t;
    Index head = 32;
    Index seq = 32;
    Index batch = 1;
    Index kv = 1;
    Index n_head_kv = 1;

    std::vector<int> dist = {0};
    TensorTraits value_traits({head, seq, batch, kv, n_head_kv},
                              {head, seq, batch, kv, n_head_kv});
    TensorTraits mask_traits({seq, seq}, {seq, seq});
    TensorTraits lse_traits({seq, batch, kv, n_head_kv},
                            {seq, batch, kv, n_head_kv});

    Tensor<T> K_a(value_traits, dist);
    Tensor<T> Q_a(value_traits, dist);
    Tensor<T> V_a(value_traits, dist);
    Tensor<T> O_a(value_traits, dist);
    Tensor<T> dO_a(value_traits, dist);
    Tensor<T> mask_a(mask_traits, dist);
    Tensor<fp32_t> lse_a(lse_traits, dist);
    Tensor<T> dK_a(value_traits, dist);
    Tensor<T> dQ_a(value_traits, dist);
    Tensor<T> dV_a(value_traits, dist);

    Tensor<T> K_b(value_traits, dist);
    Tensor<T> Q_b(value_traits, dist);
    Tensor<T> V_b(value_traits, dist);
    Tensor<T> O_b(value_traits, dist);
    Tensor<T> dO_b(value_traits, dist);
    Tensor<T> mask_b(mask_traits, dist);
    Tensor<fp32_t> lse_b(lse_traits, dist);
    Tensor<T> dK_b(value_traits, dist);
    Tensor<T> dQ_b(value_traits, dist);
    Tensor<T> dV_b(value_traits, dist);

    const Index total = head * seq * batch * kv * n_head_kv;
    std::vector<Y> host_values(total);
    for(Index i = 0; i < total; ++i)
    {
        host_values[i] = Y(0.05 * ((i % 11) - 5));
    }
    copy_vector_to_tensor(K_a, host_values);
    copy_vector_to_tensor(Q_a, host_values);
    copy_vector_to_tensor(V_a, host_values);
    copy_vector_to_tensor(K_b, host_values);
    copy_vector_to_tensor(Q_b, host_values);
    copy_vector_to_tensor(V_b, host_values);

    std::vector<Y> dO_values(total);
    for(Index i = 0; i < total; ++i)
    {
        dO_values[i] = Y(0.02 * (((i + 3) % 9) - 4));
    }
    copy_vector_to_tensor(dO_a, dO_values);
    copy_vector_to_tensor(dO_b, dO_values);

    std::vector<Y> mask_values(seq * seq);
    for(Index i = 0; i < seq; ++i)
    {
        for(Index j = 0; j < seq; ++j)
        {
            const Index idx = i * seq + j;
            if(std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 4)
            {
                mask_values[idx] = Y(0);
            }
            else
            {
                mask_values[idx] = -std::numeric_limits<Y>::infinity();
            }
        }
    }
    copy_vector_to_tensor(mask_a, mask_values);
    copy_vector_to_tensor(mask_b, mask_values);

    tensor::flash_sdpa_fwd_cudnn<T>(K_a, Q_a, mask_a, lse_a, V_a, O_a);
    tensor::flash_sdpa_fwd_cudnn<T>(K_b, Q_b, mask_b, lse_b, V_b, O_b);

    tensor::flash_sdpa_bwd_cudnn<T>(
        K_a, Q_a, V_a, O_a, dO_a, mask_a, lse_a, dK_a, dQ_a, dV_a);

    auto &K_handle = K_b.get_tile_handle(0);
    auto &Q_handle = Q_b.get_tile_handle(0);
    auto &V_handle = V_b.get_tile_handle(0);
    auto &O_handle = O_b.get_tile_handle(0);
    auto &dO_handle = dO_b.get_tile_handle(0);
    auto &mask_handle = mask_b.get_tile_handle(0);
    auto &lse_handle = lse_b.get_tile_handle(0);
    auto &dK_handle = dK_b.get_tile_handle(0);
    auto &dQ_handle = dQ_b.get_tile_handle(0);
    auto &dV_handle = dV_b.get_tile_handle(0);

    starpu::flash_sdpa_bwd_cudnn.restrict_where(STARPU_CUDA);
    starpu::flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
        seq, head, batch * kv * n_head_kv,
        K_handle, Q_handle, V_handle,
        O_handle, dO_handle,
        mask_handle, lse_handle,
        dK_handle, dQ_handle, dV_handle);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);

    auto compare = [&](const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        auto lhs_vec = tensor_to_vector(lhs);
        auto rhs_vec = tensor_to_vector(rhs);
        for(Index i = 0; i < static_cast<Index>(lhs_vec.size()); ++i)
        {
            float a = static_cast<float>(lhs_vec[i]);
            float b = static_cast<float>(rhs_vec[i]);
            const float tol = 5e-2f + 1e-2f * std::abs(a);
            TEST_ASSERT(std::abs(a - b) <= tol);
        }
    };

    compare(dK_a, dK_b);
    compare(dQ_a, dQ_b);
    compare(dV_a, dV_b);

    std::cout << "flash_sdpa_bwd_cudnn tensor test passed for "
              << T::short_name << std::endl;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    int ncpu = 1, ncuda = 1, ooc = 0, verbose = 0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    check_single_tile<fp16_t>();
    check_single_tile<bf16_t>();
    return 0;
}
