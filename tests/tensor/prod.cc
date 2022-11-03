/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/prod.cc
 * Prod operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-10-26
 * */

#include "nntile/tensor/prod.hh"
#include "nntile/tile/prod.hh"
#include "nntile/starpu/prod.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile)
{
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source and destination tensors
    TensorTraits single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src_single(single_traits, dist_root, last_tag),
        dst_single(single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        auto dst_tile = dst_single.get_tile(0);
        auto src_local = src_tile.acquire(STARPU_W);
        auto dst_local = dst_tile.acquire(STARPU_W);
        for(Index i = 0; i < src_tile.nelems; ++i)
        {
            src_local[i] = T(i);
            dst_local[i] = T(i-100);
        }
        src_local.release();
        dst_local.release();
    }
    // Generate distributed-tile source and destination tensors
    TensorTraits traits(shape, basetile);
    std::vector<int> src_distr(traits.grid.nelems), dst_distr(src_distr);
    for(Index i = 0; i < traits.grid.nelems; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
        dst_distr[i] = (i+2) % mpi_size;
    }
    Tensor<T> src(traits, src_distr, last_tag),
        dst(traits, dst_distr, last_tag);
    scatter(src_single, src);
    scatter(dst_single, dst);
    // Get prod
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        auto dst_tile = dst_single.get_tile(0);
        tile::prod<T>(src_tile, dst_tile);
    }
    prod<T>(src, dst);
    // Compare results
    Tensor<T> dst2_single(single_traits, dist_root, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile2 = dst2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < traits.nelems; ++i)
        {
            TEST_ASSERT(tile_local[i] == tile2_local[i]);
        }
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({}, {});
    check<T>({5}, {5});
    check<T>({11}, {5});
    check<T>({11, 12, 13}, {5, 6, 7});
    check<T>({1000, 1000}, {450, 450});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // No checks that throw exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::prod::init();
    starpu::subcopy::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

