/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/axpy.cc
 * AXPY operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/axpy.hh"
#include "nntile/tile/axpy.hh"
#include "nntile/starpu/axpy.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source and destination tensors
    TensorTraits single_traits(shape, shape), alpha_traits({}, {});
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src_single(single_traits, dist_root, last_tag),
        dst_single(single_traits, dist_root, last_tag),
        alpha(alpha_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto alpha_tile = alpha.get_tile(0);
        auto src_tile = src_single.get_tile(0);
        auto dst_tile = dst_single.get_tile(0);
        auto alpha_local = alpha_tile.acquire(STARPU_W);
        auto src_local = src_tile.acquire(STARPU_W);
        auto dst_local = dst_tile.acquire(STARPU_W);
        alpha_local[0] = Y(-1);
        for(Index i = 0; i < src_tile.nelems; ++i)
        {
            src_local[i] = Y(i);
            dst_local[i] = Y(i-100);
        }
        alpha_local.release();
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
    // Get axpy
    if(mpi_rank == mpi_root)
    {
        auto alpha_tile = alpha.get_tile(0);
        auto src_tile = src_single.get_tile(0);
        auto dst_tile = dst_single.get_tile(0);
        tile::axpy<T>(alpha_tile, src_tile, dst_tile);
    }
    axpy<T>(alpha, src, dst);
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
            TEST_ASSERT(Y(tile_local[i]) == Y(tile2_local[i]));
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
    starpu::axpy::init();
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::axpy::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
