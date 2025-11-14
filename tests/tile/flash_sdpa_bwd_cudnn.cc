/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/flash_sdpa_bwd_cudnn.cc
 * Flash attention SDPA backward operation on Tile<T>
 *
 * @version 1.1.0
 */

#include "nntile/context.hh"
#include "nntile/tile/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_bwd_cudnn.hh"
#include "nntile/starpu/handle.hh"
#include "../testing.hh"
#include "nntile/constants.hh"
#include <cmath>
#include <limits>
#include <iostream>

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;

    // Test parameters - use small values for testing
    Index head_size = 32;
    Index n_seq = 64;
    Index n_batch = 2;
    Index kv_group_size = 1;
    Index n_head_kv = 1;

    // Create tiles with appropriate shapes
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

    // Initialize input data
    auto K_local = K_tile.acquire(STARPU_W);
    auto Q_local = Q_tile.acquire(STARPU_W);
    auto V_local = V_tile.acquire(STARPU_W);
    auto O_local = O_tile.acquire(STARPU_W);
    auto dO_local = dO_tile.acquire(STARPU_W);

    // Fill with test values (similar to starpu test)
    for(Index i = 0; i < K_tile.nelems; ++i)
    {
        K_local[i] = T(Y(0.1 * (i % 10 - 5)));
        Q_local[i] = T(Y(0.1 * ((i + 1) % 10 - 5)));
        V_local[i] = T(Y(0.1 * ((i + 2) % 10 - 5)));
        O_local[i] = T(Y(0.1 * ((i + 4) % 10 - 5)));
        dO_local[i] = T(Y(0.1 * ((i + 3) % 10 - 5)));
    }

    auto logsumexp_local = logsumexp_tile.acquire(STARPU_W);
    for(Index i = 0; i < logsumexp_tile.nelems; ++i)
    {
        logsumexp_local[i] = fp32_t((i % 11 - 5) * 0.01f);
    }
    logsumexp_local.release();

    // Create custom mask (similar to starpu test)
    auto mask_local = mask_tile.acquire(STARPU_W);
    for(Index i = 0; i < n_seq; ++i)
    {
        for(Index j = 0; j < n_seq; ++j)
        {
            Index idx = i * n_seq + j;
            // Create a simple mask: allow attention within a window
            if (std::abs(static_cast<long>(i) - static_cast<long>(j)) <= 32)
            {
                mask_local[idx] = T(Y(0.0));  // Attend
            }
            else
            {
                mask_local[idx] = T(-std::numeric_limits<Y>::infinity());  // Mask
            }
        }
    }

    K_local.release();
    Q_local.release();
    V_local.release();
    O_local.release();
    dO_local.release();
    mask_local.release();

    // Test 1: Compare tile-level vs starpu-level backward
    {
        // Create copies for starpu-level test
        Tile<T> K_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> Q_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> V_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> O_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> dO_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> dK_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> dQ_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> dV_starpu({head_size, n_seq, n_batch, kv_group_size, n_head_kv});
        Tile<T> mask_starpu({n_seq, n_seq});
        Tile<fp32_t> logsumexp_starpu({n_seq, n_batch, kv_group_size, n_head_kv});

        // Copy data to starpu tiles
        auto K_src = K_tile.acquire(STARPU_R);
        auto Q_src = Q_tile.acquire(STARPU_R);
        auto V_src = V_tile.acquire(STARPU_R);
        auto O_src = O_tile.acquire(STARPU_R);
        auto dO_src = dO_tile.acquire(STARPU_R);
        auto mask_src = mask_tile.acquire(STARPU_R);
        auto logsumexp_src = logsumexp_tile.acquire(STARPU_R);

        auto K_dst = K_starpu.acquire(STARPU_W);
        auto Q_dst = Q_starpu.acquire(STARPU_W);
        auto V_dst = V_starpu.acquire(STARPU_W);
        auto O_dst = O_starpu.acquire(STARPU_W);
        auto dO_dst = dO_starpu.acquire(STARPU_W);
        auto mask_dst = mask_starpu.acquire(STARPU_W);
        auto logsumexp_dst = logsumexp_starpu.acquire(STARPU_W);

        for(Index i = 0; i < K_tile.nelems; ++i)
        {
            K_dst[i] = K_src[i];
            Q_dst[i] = Q_src[i];
            V_dst[i] = V_src[i];
            O_dst[i] = O_src[i];
            dO_dst[i] = dO_src[i];
        }

        for(Index i = 0; i < mask_tile.nelems; ++i)
        {
            mask_dst[i] = mask_src[i];
        }

        for(Index i = 0; i < logsumexp_tile.nelems; ++i)
        {
            logsumexp_dst[i] = logsumexp_src[i];
        }

        K_src.release();
        Q_src.release();
        V_src.release();
        O_src.release();
        dO_src.release();
        mask_src.release();
        logsumexp_src.release();

        // Initialize outputs to zero
        auto dK_starpu_local_init = dK_starpu.acquire(STARPU_W);
        auto dQ_starpu_local_init = dQ_starpu.acquire(STARPU_W);
        auto dV_starpu_local_init = dV_starpu.acquire(STARPU_W);
        for(Index i = 0; i < dK_starpu.nelems; ++i)
        {
            dK_starpu_local_init[i] = T(Y(0.0));
            dQ_starpu_local_init[i] = T(Y(0.0));
            dV_starpu_local_init[i] = T(Y(0.0));
        }
        dK_starpu_local_init.release();
        dQ_starpu_local_init.release();
        dV_starpu_local_init.release();

        K_dst.release();
        Q_dst.release();
        V_dst.release();
        O_dst.release();
        dO_dst.release();
        mask_dst.release();
        logsumexp_dst.release();

        // Call starpu-level operation
        starpu::flash_sdpa_bwd_cudnn.submit<std::tuple<T>>(
            n_seq, head_size, n_batch * kv_group_size * n_head_kv, K_starpu, Q_starpu, V_starpu,
            O_starpu, dO_starpu, mask_starpu, logsumexp_starpu,
            dK_starpu, dQ_starpu, dV_starpu);

        // Initialize tile outputs to zero
        auto dK_tile_local_init = dK_tile.acquire(STARPU_W);
        auto dQ_tile_local_init = dQ_tile.acquire(STARPU_W);
        auto dV_tile_local_init = dV_tile.acquire(STARPU_W);
        for(Index i = 0; i < dK_tile.nelems; ++i)
        {
            dK_tile_local_init[i] = T(Y(0.0));
            dQ_tile_local_init[i] = T(Y(0.0));
            dV_tile_local_init[i] = T(Y(0.0));
        }
        dK_tile_local_init.release();
        dQ_tile_local_init.release();
        dV_tile_local_init.release();

        // Call tile-level operation
        flash_sdpa_bwd_cudnn<T>(K_tile, Q_tile, V_tile, O_tile, dO_tile, mask_tile, logsumexp_tile,
                               dK_tile, dQ_tile, dV_tile);

        // Compare results
        auto dK_tile_local = dK_tile.acquire(STARPU_R);
        auto dK_starpu_local_read = dK_starpu.acquire(STARPU_R);
        auto dQ_tile_local = dQ_tile.acquire(STARPU_R);
        auto dQ_starpu_local_read = dQ_starpu.acquire(STARPU_R);
        auto dV_tile_local = dV_tile.acquire(STARPU_R);
        auto dV_starpu_local_read = dV_starpu.acquire(STARPU_R);

        Y eps = (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) ? Y(1e-2) : Y(1e-5);

        for(Index i = 0; i < dK_tile.nelems; ++i)
        {
            Y dk_tile = Y(dK_tile_local[i]);
            Y dk_starpu = Y(dK_starpu_local_read[i]);
            Y diff = std::abs(dk_tile - dk_starpu);
            Y max_val = std::max(std::abs(dk_tile), std::abs(dk_starpu));
            TEST_ASSERT(diff <= eps * (Y(1.0) + max_val));
        }

        for(Index i = 0; i < dQ_tile.nelems; ++i)
        {
            Y dq_tile = Y(dQ_tile_local[i]);
            Y dq_starpu = Y(dQ_starpu_local_read[i]);
            Y diff = std::abs(dq_tile - dq_starpu);
            Y max_val = std::max(std::abs(dq_tile), std::abs(dq_starpu));
            TEST_ASSERT(diff <= eps * (Y(1.0) + max_val));
        }

        for(Index i = 0; i < dV_tile.nelems; ++i)
        {
            Y dv_tile = Y(dV_tile_local[i]);
            Y dv_starpu = Y(dV_starpu_local_read[i]);
            Y diff = std::abs(dv_tile - dv_starpu);
            Y max_val = std::max(std::abs(dv_tile), std::abs(dv_starpu));
            TEST_ASSERT(diff <= eps * (Y(1.0) + max_val));
        }

        dK_tile_local.release();
        dK_starpu_local_read.release();
        dQ_tile_local.release();
        dQ_starpu_local_read.release();
        dV_tile_local.release();
        dV_starpu_local_read.release();

        // Unregister starpu tiles (tile-level tiles are automatically handled)
        K_starpu.unregister();
        Q_starpu.unregister();
        V_starpu.unregister();
        O_starpu.unregister();
        dO_starpu.unregister();
        dK_starpu.unregister();
        dQ_starpu.unregister();
        dV_starpu.unregister();
        mask_starpu.unregister();
        logsumexp_starpu.unregister();
    }
}

template<typename T>
void validate()
{
    // Check normal execution
    check<T>();

    // TODO: Add exception testing later
    // For now, just check that the basic functionality works

    // Tell the user that the test passed
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
