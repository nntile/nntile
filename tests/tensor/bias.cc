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
 * @date 2022-11-03
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/tile/bias.hh"
#include "nntile/starpu/bias.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"

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
    TensorTraits dst_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> dst_single(dst_single_traits, dist_root, last_tag);
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
    TensorTraits src_single_traits(src_shape, src_shape);
    Tensor<T> src_single(src_single_traits, dist_root, last_tag);
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
    Tensor<T> dst2_single(dst_single_traits, dist_root, last_tag);
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
        tile::bias<T>(val, src_tile);
    }
    bias<T>(val, src_copy_single);
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
    // Bias along the given axis
    check<T>({11}, {5}, 0);
    check<T>({11, 12}, {5, 6}, 0);
    check<T>({11, 12}, {5, 6}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 0);
    check<T>({11, 12, 13}, {5, 6, 5}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 2);
    check<T>({1000, 1000}, {450, 450}, 0);
    check<T>({1000, 1000}, {450, 450}, 1);

    // Bias all elements with the same value
    check<T>(1, {}, {});
    check<T>(10, {5}, {5});
    check<T>(-5, {11}, {5});
    check<T>(123, {11, 12, 13}, {5, 6, 7});
    check<T>(34.45, {1000, 1000}, {450, 450});

    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh3 = {3}, sh4 = {4};
    TensorTraits trA(sh34, sh23), trB(sh3, sh3), trC(sh4, sh4);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0};
    Tensor<T> A(trA, dist0000, last_tag), B(trB, dist0, last_tag),
        C(trC, dist0, last_tag);
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
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::bias::init();
    starpu::subcopy::init();
    starpu::bias::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

