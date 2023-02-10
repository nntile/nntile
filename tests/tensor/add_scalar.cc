/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/add_scalar.cc
 * Add scalar operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Katrutsa
 * @date 2023-02-10
 * */

#include "nntile/tensor/add_scalar.hh"
#include "nntile/tile/add_scalar.hh"
#include "nntile/starpu/add_scalar.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(T val, const std::vector<Index> &shape, const std::vector<Index> &basetile)
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
        src_copy_single(single_traits, dist_root, last_tag);
    
    if(mpi_rank == mpi_root)
    {
        
        auto src_tile = src_single.get_tile(0);
        auto src_copy_tile = src_copy_single.get_tile(0);
        auto src_local = src_tile.acquire(STARPU_W);
        auto src_copy_local = src_copy_tile.acquire(STARPU_W);
        for(Index i = 0; i < src_tile.nelems; ++i)
        {
            src_local[i] = T(i);
            src_copy_local[i] = T(i);
        }
        src_local.release();
        src_copy_local.release();
    }
    // Generate distributed-tile source and destination tensors
    TensorTraits traits(shape, basetile);
    std::vector<int> src_distr(traits.grid.nelems), dst_distr(src_distr);
    for(Index i = 0; i < traits.grid.nelems; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> src(traits, src_distr, last_tag);
    scatter(src_single, src);
    // Compute add_scalar
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        tile::add_scalar<T>(val, src_tile);
    }
    add_scalar<T>(val, src_copy_single);
    // Compare results
    Tensor<T> dst2_single(single_traits, dist_root, last_tag);
    gather<T>(src_copy_single, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = src_single.get_tile(0);
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
    check<T>(1, {}, {});
    check<T>(10, {5}, {5});
    check<T>(-5, {11}, {5});
    check<T>(123, {11, 12, 13}, {5, 6, 7});
    check<T>(34.45, {1000, 1000}, {450, 450});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // No checks that throw exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::add_scalar::init();
    starpu::subcopy::init();
    starpu::add_scalar::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
