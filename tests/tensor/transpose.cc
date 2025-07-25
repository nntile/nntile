/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/transpose.cc
 * Transpose operation
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/transpose.hh"
#include "nntile/starpu/transpose.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/context.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile,
        Index ndim)
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
            tile_local[i] = Y(-1);
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
    std::vector<Index> src_shape(shape.size()), src_basetile(shape.size());
    for(Index i = 0; i < shape.size(); ++i)
    {
        src_shape[(i+ndim) % shape.size()] = shape[i];
        src_basetile[(i+ndim) % shape.size()] = basetile[i];
    }
    // Generate single-tile source tensor and init it
    TensorTraits src_single_traits(src_shape, src_shape);
    Tensor<T> src_single(src_single_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        auto tile = src_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < src_single.nelems; ++i)
        {
            tile_local[i] = Y(i);
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
    Tensor<T> src(src_traits, src_distr);
    scatter<T>(src_single, src);
    // Perform multi-tile and single-tile
    transpose<T>(-1.0, src, dst, ndim);
    transpose<T>(-1.0, src_single, dst_single, ndim);
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
    // Bias along the given axis
    check<T>({11, 12}, {5, 6}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 2);
    check<T>({1000, 1000}, {450, 450}, 1);
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
//    // Check throwing exceptions
//    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh3 = {3}, sh4 = {4};
//    TensorTraits trA(sh34, sh23), trB(sh3, sh3), trC(sh4, sh4);
//    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0};
//    Tensor<T> A(trA, dist0000), B(trB, dist0),
//        C(trC, dist0);
//    TEST_THROW(transpose<T>(1.0, A, 0.0, A, 0));
//    TEST_THROW(transpose<T>(1.0, B, 0.0, A, -1));
//    TEST_THROW(transpose<T>(1.0, B, 0.0, A, 2));
//    TEST_THROW(transpose<T>(1.0, B, 0.0, A, 0));
//    TEST_THROW(transpose<T>(1.0, B, 0.0, A, 1));
//    TEST_THROW(transpose<T>(1.0, C, 0.0, A, 0));
//    TEST_THROW(transpose<T>(1.0, C, 0.0, A, 1));
}

int main(int argc, char **argv)
{
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
