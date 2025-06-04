/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/add_fiber.cc
 * Tensor wrappers for addition of a tensor and a broadcasted fiber
 *
 * @version 1.1.0
 * */

#include "nntile/context.hh"
#include "nntile/starpu/config.hh"
#include "nntile/tensor/add_fiber.hh"
#include "nntile/tile/add_fiber.hh"
#include "nntile/starpu/add_fiber.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile,
        Index axis)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile destination tensor and init it
    TensorTraits dst_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> dst_single(dst_single_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_single.nelems; ++i)
        {
            tile_local[i] = Y(i);
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
    Tensor<T> dst(dst_traits, dst_distr);
    scatter<T>(dst_single, dst);
    // Define proper shape and basetile for the source tensor
    std::vector<Index> src1_shape{shape[axis]}, src1_basetile{basetile[axis]};
    std::vector<Index> src2_shape{shape}, src2_basetile(basetile);

    // Generate single-tile source tensor and init it
    TensorTraits src1_single_traits(src1_shape, src1_shape);
    Tensor<T> src1_single(src1_single_traits, dist_root);

    TensorTraits src2_single_traits(src2_shape, src2_shape);
    Tensor<T> src2_single(src2_single_traits, dist_root);

    if(mpi_rank == mpi_root)
    {
        auto tile = src1_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < src1_single.nelems; ++i)
        {
            tile_local[i] = Y(-i);
        }
        tile_local.release();

        auto tile2 = src2_single.get_tile(0);
        auto tile2_local = tile2.acquire(STARPU_W);
        for(Index i = 0; i < src2_single.nelems; ++i)
        {
            tile2_local[i] = Y(-3*i);
        }
        tile2_local.release();
    }
    // Scatter source tensor
    TensorTraits src1_traits(src1_shape, src1_basetile);
    std::vector<int> src1_distr(src1_traits.grid.nelems);
    for(Index i = 0; i < src1_traits.grid.nelems; ++i)
    {
        src1_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> src1(src1_traits, src1_distr);
    scatter<T>(src1_single, src1);

    TensorTraits src2_traits(src2_shape, src2_basetile);
    std::vector<int> src2_distr(src2_traits.grid.nelems);
    for(Index i = 0; i < src2_traits.grid.nelems; ++i)
    {
        src2_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> src2(src2_traits, src2_distr);
    scatter<T>(src2_single, src2);
    // Perform tensor-wise and tile-wise add_fiber operations
    add_fiber<T>(-1.0, src1, 0.5, src2, dst, axis, 0);
    if(mpi_rank == mpi_root)
    {
        tile::add_fiber<T>(-1.0, src1_single.get_tile(0), 0.5,
                src2_single.get_tile(0), dst_single.get_tile(0), axis, 0);
    }
    // Compare results
    Tensor<T> dst2_single(dst_single_traits, dist_root);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile2 = dst2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < dst_traits.nelems; ++i)
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
    check<T>({11}, {5}, 0);
    check<T>({11, 12}, {5, 6}, 0);
    check<T>({11, 12}, {5, 6}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 0);
    check<T>({11, 12, 13}, {5, 6, 5}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 2);
    check<T>({1000, 1000}, {450, 450}, 0);
    check<T>({1000, 1000}, {450, 450}, 1);
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh3 = {3}, sh4 = {4};
    TensorTraits trA(sh34, sh23), trB(sh3, sh3), trC(sh4, sh4);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0};
    Tensor<T> A(trA, dist0000), B(trB, dist0), C(trC, dist0);
    TEST_THROW(add_fiber<T>(1.0, A, 0.0, A, C, 0, 0));
    TEST_THROW(add_fiber<T>(1.0, B, 0.0, A, C, -1, 0));
    TEST_THROW(add_fiber<T>(1.0, B, 0.0, A, B, 2, 0));
    TEST_THROW(add_fiber<T>(1.0, B, 0.0, A, B, 0, 0));
    TEST_THROW(add_fiber<T>(1.0, B, 0.0, A, A, 1, 0));
    TEST_THROW(add_fiber<T>(1.0, C, 0.0, A, A, 0, 0));
    TEST_THROW(add_fiber<T>(1.0, C, 0.0, A, B, 1, 0));
}

int main(int argc, char **argv)
{
    // Initialize StarPU
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();

    return 0;
}
