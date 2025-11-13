/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/flash_sdpa_fwd_cudnn.cc
 * Flash attention SDPA forward operation on Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/tensor/flash_sdpa_fwd_cudnn.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tile/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/flash_sdpa_fwd_cudnn.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"
#include <cmath>
#include <limits>
#include <iostream>
#include <type_traits>

using namespace nntile;
using namespace nntile::tensor;

struct FlashTensorCase
{
    Index head_size;
    Index head_size_tile;
    Index n_seq;
    Index n_seq_tile;
    Index n_batch;
    Index batch_tile;
    Index kv_group_size;
    Index kv_group_tile;
    Index n_head_kv;
    Index head_kv_tile;
};

template<typename T>
void check(const FlashTensorCase &cfg)
{
    using Y = typename T::repr_t;

    // Define tensor shapes for the current configuration
    Index head_size = cfg.head_size;
    Index head_size_tile = cfg.head_size_tile;
    Index n_seq = cfg.n_seq;
    Index n_seq_tile = cfg.n_seq_tile;
    Index n_batch = cfg.n_batch;
    Index batch_tile = cfg.batch_tile;
    Index kv_group_size = cfg.kv_group_size;
    Index kv_group_tile = cfg.kv_group_tile;
    Index n_head_kv = cfg.n_head_kv;
    Index head_kv_tile = cfg.head_kv_tile;

    // Create single-tile tensors (assuming one tile per tensor as requested)
    std::vector<Index> K_shape = {head_size, n_seq, n_batch, kv_group_size, n_head_kv};
    std::vector<Index> logsumexp_shape = {n_seq, n_batch, kv_group_size, n_head_kv};

    TensorTraits K_traits(K_shape, K_shape);
    TensorTraits Q_traits(K_shape, K_shape);
    TensorTraits V_traits(K_shape, K_shape);
    TensorTraits A_traits(K_shape, K_shape);
    TensorTraits mask_traits({n_seq, n_seq}, {n_seq, n_seq});
    TensorTraits logsumexp_traits(logsumexp_shape, logsumexp_shape);

    // Use root rank for all tensors (single MPI rank scenario)
    int mpi_root = 0;
    std::vector<int> dist_root = {mpi_root};

    Tensor<T> K_single(K_traits, dist_root);
    Tensor<T> Q_single(Q_traits, dist_root);
    Tensor<T> V_single(V_traits, dist_root);
    Tensor<T> A_single(A_traits, dist_root);
    Tensor<T> mask_single(mask_traits, dist_root);
    Tensor<fp32_t> logsumexp_single(logsumexp_traits, dist_root);

    int mpi_rank = starpu_mpi_world_rank();

    if(mpi_rank == mpi_root)
    {
        // Initialize input data using tile operations
        auto K_tile = K_single.get_tile(0);
        auto Q_tile = Q_single.get_tile(0);
        auto V_tile = V_single.get_tile(0);
        auto A_tile = A_single.get_tile(0);
        auto mask_tile = mask_single.get_tile(0);
        auto logsumexp_tile = logsumexp_single.get_tile(0);

        auto K_local = K_tile.acquire(STARPU_W);
        auto Q_local = Q_tile.acquire(STARPU_W);
        auto V_local = V_tile.acquire(STARPU_W);
        auto A_local = A_tile.acquire(STARPU_W);
        auto mask_local = mask_tile.acquire(STARPU_W);
        auto logsumexp_local = logsumexp_tile.acquire(STARPU_W);

        // Fill with test values (similar to starpu test)
        for(Index i = 0; i < K_tile.nelems; ++i)
        {
            K_local[i] = T(Y(0.1 * (i % 10 - 5)));
            Q_local[i] = T(Y(0.1 * ((i + 1) % 10 - 5)));
            V_local[i] = T(Y(0.1 * ((i + 2) % 10 - 5)));
            A_local[i] = T(Y(0.0)); // Initialize output to zero
        }

        for(Index i = 0; i < logsumexp_tile.nelems; ++i)
        {
            logsumexp_local[i] = -std::numeric_limits<float>::infinity();
        }

        // Create custom mask (similar to starpu test)
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
        A_local.release();
        mask_local.release();
        logsumexp_local.release();
    }

    // Build multi-tile tensors and scatter data
    auto make_distribution = [](Index nelems)
    {
        std::vector<int> distr(nelems);
        for(Index i = 0; i < nelems; ++i)
        {
            distr[i] = static_cast<int>((i % 2));
        }
        return distr;
    };

    // Base tile shapes make it explicit what each dimension stands for
    std::vector<Index> kv_tensor_tile_shape = {
        head_size_tile,    // head_size tile extent
        n_seq_tile,        // sequence length tile extent
        batch_tile,        // batch tile extent
        kv_group_tile,     // kv_group_size tile extent
        head_kv_tile       // n_head_kv tile extent
    };
    std::vector<Index> mask_tile_shape = {
        n_seq_tile,        // row sequence tile extent
        n_seq_tile         // col sequence tile extent
    };
    std::vector<Index> logsumexp_tile_shape = {
        n_seq_tile,        // sequence tile extent
        batch_tile,        // batch tile extent
        kv_group_tile,     // kv_group_size tile extent
        head_kv_tile       // n_head_kv tile extent
    };

    TensorTraits K_multi_traits(K_shape, kv_tensor_tile_shape);
    TensorTraits Q_multi_traits(K_shape, kv_tensor_tile_shape);
    TensorTraits V_multi_traits(K_shape, kv_tensor_tile_shape);
    TensorTraits A_multi_traits(K_shape, kv_tensor_tile_shape);
    TensorTraits mask_multi_traits({n_seq, n_seq}, mask_tile_shape);
    TensorTraits logsumexp_multi_traits(logsumexp_shape, logsumexp_tile_shape);

    auto K_distr = make_distribution(K_multi_traits.grid.nelems);
    auto Q_distr = make_distribution(Q_multi_traits.grid.nelems);
    auto V_distr = make_distribution(V_multi_traits.grid.nelems);
    auto A_distr = make_distribution(A_multi_traits.grid.nelems);
    auto mask_distr = make_distribution(mask_multi_traits.grid.nelems);
    auto logsumexp_distr = make_distribution(logsumexp_multi_traits.grid.nelems);

    Tensor<T> K_multi(K_multi_traits, K_distr);
    Tensor<T> Q_multi(Q_multi_traits, Q_distr);
    Tensor<T> V_multi(V_multi_traits, V_distr);
    Tensor<T> A_multi(A_multi_traits, A_distr);
    Tensor<T> mask_multi(mask_multi_traits, mask_distr);
    Tensor<fp32_t> logsumexp_multi(logsumexp_multi_traits, logsumexp_distr);

    scatter<T>(K_single, K_multi);
    scatter<T>(Q_single, Q_multi);
    scatter<T>(V_single, V_multi);
    scatter<T>(A_single, A_multi);
    scatter<T>(mask_single, mask_multi);
    scatter<fp32_t>(logsumexp_single, logsumexp_multi);

    // Perform tensor-wise and tile-wise flash_sdpa_fwd_cudnn operations
    if(mpi_rank == mpi_root)
    {
        // Call tile-level operation (reference)
        tile::flash_sdpa_fwd_cudnn<T>(K_single.get_tile(0), Q_single.get_tile(0),
                                     mask_single.get_tile(0), logsumexp_single.get_tile(0),
                                     V_single.get_tile(0), A_single.get_tile(0));
    }

    // Call tensor-level operation
    flash_sdpa_fwd_cudnn<T>(K_multi, Q_multi, mask_multi, logsumexp_multi,
                           V_multi, A_multi);

    // Compare results
    Tensor<T> A_multi_gather(A_traits, dist_root);
    Tensor<fp32_t> logsumexp_multi_gather(logsumexp_traits, dist_root);
    gather<T>(A_multi, A_multi_gather);
    gather<fp32_t>(logsumexp_multi, logsumexp_multi_gather);

    if(mpi_rank == mpi_root)
    {
        auto A_ref_tile = A_single.get_tile(0);
        auto A_multi_tile = A_multi_gather.get_tile(0);
        auto logsumexp_ref_tile = logsumexp_single.get_tile(0);
        auto logsumexp_multi_tile = logsumexp_multi_gather.get_tile(0);

        auto A_ref_local = A_ref_tile.acquire(STARPU_R);
        auto A_multi_local = A_multi_tile.acquire(STARPU_R);
        auto logsumexp_ref_local = logsumexp_ref_tile.acquire(STARPU_R);
        auto logsumexp_multi_local = logsumexp_multi_tile.acquire(STARPU_R);

        Y eps = (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) ? Y(1e-2) : Y(1e-5);
        const float eps_fp32 = (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) ? 1e-2f : 1e-5f;

        for(Index i = 0; i < logsumexp_single.nelems; ++i)
        {
            float ref_val = static_cast<float>(logsumexp_ref_local[i]);
            float multi_val = static_cast<float>(logsumexp_multi_local[i]);
            float diff = std::abs(ref_val - multi_val);
            float max_val = std::max(std::abs(ref_val), std::abs(multi_val));
            TEST_ASSERT(diff <= eps_fp32 * (1.0f + max_val));
        }

        for(Index i = 0; i < A_single.nelems; ++i)
        {
            Y ref_val = Y(A_ref_local[i]);
            Y multi_val = Y(A_multi_local[i]);
            Y diff = std::abs(ref_val - multi_val);
            Y max_val = std::max(std::abs(ref_val), std::abs(multi_val));
            TEST_ASSERT(diff <= eps * (Y(1.0) + max_val));
        }

        A_ref_local.release();
        A_multi_local.release();
        logsumexp_ref_local.release();
        logsumexp_multi_local.release();
    }
}

template<typename T>
void validate()
{
    // Order: head_size, head_size_tile, n_seq, n_seq_tile, n_batch, batch_tile,
    //        kv_group_size, kv_group_tile, n_head_kv, head_kv_tile.
    // head_size_tile must be equal to head_size (no parallelism across head_size)
    check<T>({32, 32, 32, 32, 1, 1, 1, 1, 1, 1});
    check<T>({32, 32, 64, 32, 1, 1, 1, 1, 1, 1});
    check<T>({64, 64, 64, 16, 3, 1, 2, 1, 4, 2});

    // TODO: Add exception testing later
    // For now, just check that the basic functionality works

    // Tell the user that the test passed
    std::cout << "flash_sdpa_fwd_cudnn<" << T::short_name << "> passed\n";
}

int main(int argc, char **argv)
{
    // Initialize StarPU
    int ncpu=1, ncuda=1, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Only test FP16 and BF16 as per cuDNN limitations
    validate<fp16_t>();
    validate<bf16_t>();

    return 0;
}
