/*! @file tests/tile/flash_sdpa_bwd_cudnn.cc
 * Flash attention SDPA backward operation on Tile<T>
 */

#include "nntile/context.hh"
#include "nntile/tile/flash_sdpa_fwd_cudnn.hh"
#include "nntile/tile/flash_sdpa_bwd_cudnn.hh"
#include "../testing.hh"

#include <cmath>
#include <iostream>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;

    const Index head_size = 32;
    const Index n_seq = 32;
    const Index n_batch = 1;
    const Index kv_group_size = 1;
    const Index n_head_kv = 1;

    Tile<T> K_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> Q_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> V_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> O_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> dO_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> dK_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> dQ_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> dV_tile({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
    Tile<T> mask_tile({n_seq, n_seq});
    Tile<fp32_t> logsumexp_tile({n_seq, n_batch, kv_group_size, n_head_kv});

    auto fill_with_pattern = [](const Tile<T> &tile,
            TileLocalData<T> &local, Y scale) {
        for(Index i = 0; i < tile.nelems; ++i)
        {
            local[i] = T(scale * Y((i % 13) - 6));
        }
    };

    auto K_local = K_tile.acquire(STARPU_W);
    auto Q_local = Q_tile.acquire(STARPU_W);
    auto V_local = V_tile.acquire(STARPU_W);
    auto O_local = O_tile.acquire(STARPU_W);
    fill_with_pattern(K_tile, K_local, Y(0.1));
    fill_with_pattern(Q_tile, Q_local, Y(0.08));
    fill_with_pattern(V_tile, V_local, Y(0.06));
    fill_with_pattern(O_tile, O_local, Y(0.0));
    K_local.release();
    Q_local.release();
    V_local.release();
    O_local.release();

    auto mask_local = mask_tile.acquire(STARPU_W);
    for(Index i = 0; i < n_seq; ++i)
    {
        for(Index j = 0; j < n_seq; ++j)
        {
            const Index idx = i * n_seq + j;
            if(std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 4)
            {
                mask_local[idx] = T(Y(0.0));
            }
            else
            {
                mask_local[idx] = T(-std::numeric_limits<Y>::infinity());
            }
        }
    }
    mask_local.release();

    auto lse_local = logsumexp_tile.acquire(STARPU_W);
    for(Index i = 0; i < logsumexp_tile.nelems; ++i)
    {
        lse_local[i] = fp32_t(-std::numeric_limits<float>::infinity());
    }
    lse_local.release();

    // Run forward once to populate O and stats
    flash_sdpa_fwd_cudnn<T>(
        K_tile, Q_tile, mask_tile, logsumexp_tile, V_tile, O_tile);
    starpu_task_wait_for_all();

    // Test 1: zero upstream gradient should produce zero downstream grads
    auto dO_local = dO_tile.acquire(STARPU_W);
    auto dK_local = dK_tile.acquire(STARPU_W);
    auto dQ_local = dQ_tile.acquire(STARPU_W);
    auto dV_local = dV_tile.acquire(STARPU_W);
    for(Index i = 0; i < dO_tile.nelems; ++i)
    {
        dO_local[i] = T(Y(0));
    }
    for(Index i = 0; i < dK_tile.nelems; ++i)
    {
        dK_local[i] = T(Y(0));
        dQ_local[i] = T(Y(0));
        dV_local[i] = T(Y(0));
    }
    dO_local.release();
    dK_local.release();
    dQ_local.release();
    dV_local.release();

    flash_sdpa_bwd_cudnn<T>(
        K_tile, Q_tile, V_tile, O_tile, dO_tile, mask_tile, logsumexp_tile,
        dK_tile, dQ_tile, dV_tile);
    starpu_task_wait_for_all();

    auto check_zero = [](auto &tile) {
        auto local = tile.acquire(STARPU_R);
        for(Index i = 0; i < tile.nelems; ++i)
        {
            Y val = static_cast<Y>(local[i]);
            TEST_ASSERT(std::abs(val) < Y(1e-3));
        }
        local.release();
    };
    check_zero(dK_tile);
    check_zero(dQ_tile);
    check_zero(dV_tile);

    // Test 2: random upstream gradient leads to finite downstream gradients
    dO_local = dO_tile.acquire(STARPU_W);
    for(Index i = 0; i < dO_tile.nelems; ++i)
    {
        dO_local[i] = T(Y(0.05 * ((i % 7) - 3)));
    }
    dO_local.release();

    dK_local = dK_tile.acquire(STARPU_W);
    dQ_local = dQ_tile.acquire(STARPU_W);
    dV_local = dV_tile.acquire(STARPU_W);
    for(Index i = 0; i < dK_tile.nelems; ++i)
    {
        dK_local[i] = T(Y(0));
        dQ_local[i] = T(Y(0));
        dV_local[i] = T(Y(0));
    }
    dK_local.release();
    dQ_local.release();
    dV_local.release();

    flash_sdpa_bwd_cudnn<T>(
        K_tile, Q_tile, V_tile, O_tile, dO_tile, mask_tile, logsumexp_tile,
        dK_tile, dQ_tile, dV_tile);
    starpu_task_wait_for_all();

    auto check_finite = [](auto &tile) {
        auto local = tile.acquire(STARPU_R);
        bool non_zero = false;
        for(Index i = 0; i < tile.nelems; ++i)
        {
            Y val = static_cast<Y>(local[i]);
            TEST_ASSERT(std::isfinite(static_cast<double>(val)));
            non_zero |= std::abs(val) > Y(1e-5);
        }
        TEST_ASSERT(non_zero);
        local.release();
    };
    check_finite(dK_tile);
    check_finite(dQ_tile);
    check_finite(dV_tile);
}

template<typename T>
void validate()
{
    check<T>();
    std::cout << "flash_sdpa_bwd_cudnn<" << T::short_name << "> passed\n";
}

int main(int argc, char **argv)
{
    int ncpu = 1, ncuda = 1, ooc = 0, verbose = 0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    validate<fp16_t>();
    validate<bf16_t>();

    return 0;
}
