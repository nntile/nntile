/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/multiply_slice_inplace.cc
 * Test for tensor::multiply_slice_inplace<T> C++ wrapper
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/multiply_slice_inplace.hh"
#include "nntile/tile/multiply_slice_inplace.hh"
#include "nntile/starpu/multiply_slice_inplace.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "nntile/context.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void test_multiply_slice_inplace()
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    const Y alpha{1.5}, beta{0.5};
    // Set up shapes and axis
    std::vector<Index> shape_src{2, 3}, shape_dst{2, 4, 3};
    Index axis{1};
    // Generate single-tile source and destination tensors
    TensorTraits single_traits(shape_dst, shape_dst);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src_single(single_traits, dist_root),
        dst_single(single_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        auto dst_tile = dst_single.get_tile(0);
        auto src_local = src_tile.acquire(STARPU_W);
        auto dst_local = dst_tile.acquire(STARPU_W);
        for(Index i = 0; i < src_tile.nelems; ++i)
        {
            src_local[i] = Y(i);
        }
        for(Index i = 0; i < dst_tile.nelems; ++i)
        {
            dst_local[i] = Y(i + 100);
        }
        src_local.release();
        dst_local.release();
    }
    // Set up proper shapes for source and destination tensors
    std::vector<Index> basetile{2, 2, 3};
    // Scatter source and destination tensors
    TensorTraits src_traits(shape_src, basetile);
    TensorTraits dst_traits(shape_dst, basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = i % mpi_size;
    }
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = i % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr), dst(dst_traits, dst_distr);
    scatter<T>(src_single, src);
    scatter<T>(dst_single, dst);
    // Perform tensor-wise and tile-wise multiply_slice_inplace operations
    multiply_slice_inplace<T>(alpha, src, beta, dst, axis);
    // Check results for the first tile
    auto tile_src = src.get_tile(0);
    auto tile_dst = dst.get_tile(0);
    tile::multiply_slice_inplace<T>(alpha, tile_src, beta, tile_dst, axis);
    // Check result
    std::vector<Y> result_dst(dst.nelems);
    if(mpi_rank == mpi_root)
    {
        auto tile_dst = dst.get_tile(0);
        auto tile_local = tile_dst.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            result_dst[i] = Y(tile_local[i]);
        }
        tile_local.release();
    }
    // Check if result is correct
    for(Index i0 = 0; i0 < shape_dst[0]; ++i0)
    {
        for(Index i1 = 0; i1 < shape_dst[1]; ++i1)
        {
            for(Index i2 = 0; i2 < shape_dst[2]; ++i2)
            {
                Index linear = i0*shape_dst[1]*shape_dst[2] + i1*shape_dst[2] + i2;
                Y expected = beta * Y(i0*shape_dst[1]*shape_dst[2] + i1*shape_dst[2] + i2 + 100) * alpha * Y(i0*shape_src[1] + i2);
                TEST_ASSERT(std::abs(Y{result_dst[linear]} - expected) < 1e-5);
            }
        }
    }
}

template<typename T>
void test_multiply_slice_inplace_async()
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    const Y alpha{2.0}, beta{0.25};
    // Set up shapes and axis
    std::vector<Index> shape_src{3, 4}, shape_dst{3, 5, 4};
    Index axis{1};
    // Generate single-tile source and destination tensors
    TensorTraits single_traits(shape_dst, shape_dst);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src_single(single_traits, dist_root),
        dst_single(single_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        auto dst_tile = dst_single.get_tile(0);
        auto src_local = src_tile.acquire(STARPU_W);
        auto dst_local = dst_tile.acquire(STARPU_W);
        for(Index i = 0; i < src_tile.nelems; ++i)
        {
            src_local[i] = Y(i);
        }
        for(Index i = 0; i < dst_tile.nelems; ++i)
        {
            dst_local[i] = Y(i + 200);
        }
        src_local.release();
        dst_local.release();
    }
    // Set up proper shapes for source and destination tensors
    std::vector<Index> basetile{3, 3, 4};
    // Scatter source and destination tensors
    TensorTraits src_traits(shape_src, basetile);
    TensorTraits dst_traits(shape_dst, basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = i % mpi_size;
    }
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = i % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr), dst(dst_traits, dst_distr);
    scatter<T>(src_single, src);
    scatter<T>(dst_single, dst);
    // Perform tensor-wise and tile-wise multiply_slice_inplace operations
    multiply_slice_inplace_async<T>(alpha, src, beta, dst, axis);
    starpu_task_wait_for_all();
    // Check results for the first tile
    auto tile_src = src.get_tile(0);
    auto tile_dst = dst.get_tile(0);
    tile::multiply_slice_inplace_async<T>(alpha, tile_src, beta, tile_dst, axis);
    starpu_task_wait_for_all();
    // Check result
    std::vector<Y> result_dst(dst.nelems);
    if(mpi_rank == mpi_root)
    {
        auto tile_dst = dst.get_tile(0);
        auto tile_local = tile_dst.acquire(STARPU_R);
        for(Index i = 0; i < dst.nelems; ++i)
        {
            result_dst[i] = Y(tile_local[i]);
        }
        tile_local.release();
    }
    // Check if result is correct
    for(Index i0 = 0; i0 < shape_dst[0]; ++i0)
    {
        for(Index i1 = 0; i1 < shape_dst[1]; ++i1)
        {
            for(Index i2 = 0; i2 < shape_dst[2]; ++i2)
            {
                Index linear = i0*shape_dst[1]*shape_dst[2] + i1*shape_dst[2] + i2;
                Y expected = beta * Y(i0*shape_dst[1]*shape_dst[2] + i1*shape_dst[2] + i2 + 200) * alpha * Y(i0*shape_src[1] + i2);
                TEST_ASSERT(std::abs(Y{result_dst[linear]} - expected) < 1e-5);
            }
        }
    }
}

template<typename T>
void test_multiply_slice_inplace_errors()
{
    using Y = typename T::repr_t;
    // Create tensor traits with wrong shapes for error testing
    std::vector<Index> sh23 = {2, 3}, sh243 = {2, 4, 3}, sh245 = {2, 4, 5};
    TensorTraits trA(sh23, sh23), trB(sh243, sh243), trC(sh23, sh23),
                 trD(sh245, sh245), trE(sh243, sh243), trF(sh23, sh23), trG(sh243, sh243);
    std::vector<int> dist0 = {0};
    Tensor<T> A(trA, dist0), B(trB, dist0), C(trC, dist0),
              D(trD, dist0), E(trE, dist0), F(trF, dist0), G(trG, dist0);
    // Test various error conditions
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, F, 1.0, F, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, -1));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 2));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, D, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, E, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, B, 1));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, G, 0));
    TEST_THROW(multiply_slice_inplace<T>(1.0, A, 1.0, G, 1));
}

int main(int argc, char **argv)
{
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    test_multiply_slice_inplace<fp32_t>();
    test_multiply_slice_inplace<fp64_t>();
    test_multiply_slice_inplace<bf16_t>();
    test_multiply_slice_inplace<fp16_t>();

    test_multiply_slice_inplace_async<fp32_t>();
    test_multiply_slice_inplace_async<fp64_t>();
    test_multiply_slice_inplace_async<bf16_t>();
    test_multiply_slice_inplace_async<fp16_t>();

    test_multiply_slice_inplace_errors<fp32_t>();
    test_multiply_slice_inplace_errors<fp64_t>();
    test_multiply_slice_inplace_errors<bf16_t>();
    test_multiply_slice_inplace_errors<fp16_t>();

    return 0;
}
