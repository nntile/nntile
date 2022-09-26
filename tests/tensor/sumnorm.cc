/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/sumnorm.cc
 * Sumnorm operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-26
 * */

#include "nntile/tensor/sumnorm.hh"
#include "nntile/tile/sumnorm.hh"
#include "nntile/tile/clear.hh"
#include "nntile/starpu/sumnorm.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/clear.hh"
#include "../testing.hh"
#include <limits>

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile,
        Index axis)
{
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source tensor and init it
    Tensor<T> src_single({shape, shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = src_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_single.nelems; ++i)
        {
            tile_local[i] = T(i);
        }
        tile_local.release();
    }
    // Scatter source tensor
    TensorTraits src_traits(shape, basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr, last_tag);
    scatter<T>(src_single, src);
    // Define proper shape and basetile for the dest tensor
    std::vector<Index> dst_shape(shape), dst_basetile(basetile);
    dst_shape[0] = 2;
    dst_basetile[0] = 2;
    for(Index i = 1; i <= axis; ++i)
    {
        dst_shape[i] = shape[i-1];
        dst_basetile[i] = basetile[i-1];
    }
    // Generate single-tile and distributed dest tensors
    Tensor<T> dst_single({dst_shape, dst_shape}, {mpi_root}, last_tag);
    TensorTraits dst_traits(dst_shape, dst_basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    // Perform tensor-wise and tile-wise sumnorm operations
    sumnorm<T>(src, dst, axis);
    if(mpi_rank == mpi_root)
    {
        tile::clear(dst_single.get_tile(0));
        tile::sumnorm<T>(src_single.get_tile(0), dst_single.get_tile(0), axis);
    }
    // Compare results
    Tensor<T> dst2_single({dst_shape, dst_shape}, {mpi_root}, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile2 = dst2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < dst_traits.nelems; ++i)
        {
            T diff = std::abs(tile_local[i] - tile2_local[i]);
            T abs = std::abs(tile_local[i]);
            TEST_ASSERT(diff/abs < 10*std::numeric_limits<T>::epsilon());
        }
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({11}, {5}, 0);
    check<T>({11, 12}, {5, 6}, 0);
    check<T>({11, 12}, {5, 6}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 0);
    check<T>({11, 12, 13}, {5, 6, 5}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 2);
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    Tensor<T> A({{3, 4}, {2, 3}}, {0, 0, 0, 0}, last_tag),
        B({{2, 3}, {2, 3}}, {0}, last_tag),
        C({{4}, {4}}, {0}, last_tag),
        D({{3, 3}, {2, 3}}, {0, 0}, last_tag),
        E({{2, 4}, {1, 3}}, {0, 0, 0, 0}, last_tag),
        F({{}, {}}, {0}, last_tag),
        G({{2, 4}, {2, 2}}, {0, 0}, last_tag);
    TEST_THROW(sumnorm<T>(A, C, 0));
    TEST_THROW(sumnorm<T>(F, F, 0));
    TEST_THROW(sumnorm<T>(A, B, -1));
    TEST_THROW(sumnorm<T>(A, B, 2));
    TEST_THROW(sumnorm<T>(A, D, 0));
    TEST_THROW(sumnorm<T>(A, E, 0));
    TEST_THROW(sumnorm<T>(A, B, 0));
    TEST_THROW(sumnorm<T>(A, B, 1));
    TEST_THROW(sumnorm<T>(A, G, 0));
    TEST_THROW(sumnorm<T>(A, G, 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::sumnorm::init();
    starpu::subcopy::init();
    starpu::clear::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

