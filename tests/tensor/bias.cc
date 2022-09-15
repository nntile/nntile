/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/bias.cc
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-15
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/tile/bias.hh"
#include "nntile/starpu/bias.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

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
    // Generate single-tile destination tensor and init it
    Tensor<T> dst_single({shape, shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_single.nelems; ++i)
        {
            tile_local[i] = T(i);
        }
        tile_local.release();
    }
    // Scatter destination tensor
    TensorTraits dst_traits(shape, basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    scatter<T>(dst_single, dst);
    // Define proper shape and basetile for the source tensor
    std::vector<Index> src_shape(dst_traits.ndim-1),
        src_basetile(dst_traits.ndim-1);
    for(Index i = 0; i < axis; ++i)
    {
        src_shape[i] = shape[i];
        src_basetile[i] = basetile[i];
    }
    for(Index i = axis+1; i < dst_traits.ndim; ++i)
    {
        src_shape[i-1] = shape[i];
        src_basetile[i-1] = basetile[i];
    }
    // Generate single-tile source tensor and init it
    Tensor<T> src_single({src_shape, src_shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = src_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_single.nelems; ++i)
        {
            tile_local[i] = T(-i);
        }
        tile_local.release();
    }
    // Scatter source tensor
    TensorTraits src_traits(src_shape, src_basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr, last_tag);
    scatter<T>(src_single, src);
    // Perform tensor-wise and tile-wise bias operations
    bias<T>(src, dst, axis);
    if(mpi_rank == mpi_root)
    {
        tile::bias<T>(src_single.get_tile(0), dst_single.get_tile(0), axis);
    }
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
        B({{3}, {3}}, {0}, last_tag), C({{4}, {4}}, {0}, last_tag);
    TEST_THROW(bias<T>(A, A, 0));
    TEST_THROW(bias<T>(B, A, -1));
    TEST_THROW(bias<T>(B, A, 2));
    TEST_THROW(bias<T>(B, A, 0));
    TEST_THROW(bias<T>(B, A, 1));
    TEST_THROW(bias<T>(C, A, 0));
    TEST_THROW(bias<T>(C, A, 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::bias::init();
    starpu::subcopy::init();
    // Restrict execution to CPU to properly compare results
    starpu::bias::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

