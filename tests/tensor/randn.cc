/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/randn.cc
 * Randn operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-13
 * */

#include "nntile/tensor/randn.hh"
#include "nntile/tile/randn.hh"
#include "nntile/starpu/randn.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

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
    unsigned long long seed = -1;
    T mean = -1;
    T stddev = 2;
    std::vector<Index> start(shape.size());
    // Generate single-tile destination tensor
    Tensor<T> dst_single({shape, shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        tile::randn<T>(dst_single.get_tile(0), start, shape, seed, mean,
                stddev);
    }
    // Generate distributed-tile destination tensor
    TensorTraits dst_traits(shape, basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    randn<T>(dst, start, shape, seed, mean, stddev);
    // Compare results
    Tensor<T> dst2_single({shape, shape}, {mpi_root}, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile2 = dst2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < dst_traits.nelems; ++i)
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
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::randn::init();
    starpu::subcopy::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

