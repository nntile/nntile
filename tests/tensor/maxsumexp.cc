/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/maxsumexp.cc
 * maxsumexp operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/maxsumexp.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tile/maxsumexp.hh"
#include "nntile/tile/clear.hh"
#include "nntile/starpu/maxsumexp.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/context.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"
#include <limits>

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
    // Generate single-tile source tensor and init it
    TensorTraits src_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
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
    TensorTraits src_traits(shape, basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr);
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
    TensorTraits dst_single_traits(dst_shape, dst_shape);
    Tensor<T> dst_single(dst_single_traits, dist_root);
    TensorTraits dst_traits(dst_shape, dst_basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr);
    // Perform tensor-wise and tile-wise maxsumexp operations
    clear<T>(dst);
    maxsumexp<T>(src, dst, axis);
    if(mpi_rank == mpi_root)
    {
        tile::clear(dst_single.get_tile(0));
        tile::maxsumexp<T>(src_single.get_tile(0), dst_single.get_tile(0), axis);
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
            Y diff = std::abs(Y(tile_local[i]) - Y(tile2_local[i]));
            Y abs = std::abs(Y(tile_local[i]));
            TEST_ASSERT(diff/abs < 10*T::epsilon);
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
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh4 = {4}, sh33 = {3, 3},
        sh24 = {2, 4}, sh13 = {1, 3}, sh_ = {}, sh22 = {2, 2};
    TensorTraits trA(sh34, sh23), trB(sh23, sh23), trC(sh4, sh4),
        trD(sh33, sh23), trE(sh24, sh13), trF(sh_, sh_), trG(sh24, sh22);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0}, dist00 = {0, 0};
    Tensor<T> A(trA, dist0000), B(trB, dist0),
        C(trC, dist0), D(trD, dist00),
        E(trE, dist0000), F(trF, dist0),
        G(trG, dist00);
    TEST_THROW(maxsumexp<T>(A, C, 0));
    TEST_THROW(maxsumexp<T>(F, F, 0));
    TEST_THROW(maxsumexp<T>(A, B, -1));
    TEST_THROW(maxsumexp<T>(A, B, 2));
    TEST_THROW(maxsumexp<T>(A, D, 0));
    TEST_THROW(maxsumexp<T>(A, E, 0));
    TEST_THROW(maxsumexp<T>(A, B, 0));
    TEST_THROW(maxsumexp<T>(A, B, 1));
    TEST_THROW(maxsumexp<T>(A, G, 0));
    TEST_THROW(maxsumexp<T>(A, G, 1));
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
