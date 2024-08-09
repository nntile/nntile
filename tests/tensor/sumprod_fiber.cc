/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/sumprod_fiber.cc
 * Sums over fibers into a slice of a product of two Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/sumprod_fiber.hh"
#include "nntile/tile/sumprod_fiber.hh"
#include "nntile/tile/clear.hh"
#include "nntile/starpu/sumprod_fiber.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/clear.hh"
#include "../testing.hh"
#include <limits>

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(Scalar alpha, Scalar beta, const std::vector<Index> &shape,
        const std::vector<Index> &basetile, Index axis)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source tensor and init it
    TensorTraits src_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src1_single(src_single_traits, dist_root, last_tag);
    Tensor<T> src2_single(src_single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile1 = src1_single.get_tile(0);
        auto tile1_local = tile1.acquire(STARPU_W);
        auto tile2 = src2_single.get_tile(0);
        auto tile2_local = tile2.acquire(STARPU_W);
        for(Index i = 0; i < src1_single.nelems; ++i)
        {
            tile1_local[i] = Y((i+1)*(i+2));
            tile2_local[i] = 1.0 / Y(i+1);
        }
        tile1_local.release();
        tile2_local.release();
    }
    // Scatter source tensor
    TensorTraits src_traits(shape, basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> src1(src_traits, src_distr, last_tag);
    Tensor<T> src2(src_traits, src_distr, last_tag);
    scatter<T>(src1_single, src1);
    scatter<T>(src2_single, src2);
    // Define proper shape and basetile for the dest tensor
    std::vector<Index> dst_shape{shape[axis]},
        dst_basetile{basetile[axis]};
    // Generate single-tile and distributed dest tensors
    TensorTraits dst_single_traits(dst_shape, dst_shape);
    Tensor<T> dst_single(dst_single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile1 = dst_single.get_tile(0);
        auto tile1_local = tile1.acquire(STARPU_W);
        for(Index i = 0; i < dst_single.nelems; ++i)
        {
            tile1_local[i] = Y(1.0);
        }
        tile1_local.release();
    }
    TensorTraits dst_traits(dst_shape, dst_basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    scatter<T>(dst_single, dst);
    // Perform tensor-wise and tile-wise sumprod_fiber operations
    sumprod_fiber<T>(alpha, src1, src2, beta, dst, axis);
    if(mpi_rank == mpi_root)
    {
        tile::sumprod_fiber<T>(alpha, src1_single.get_tile(0),
                src2_single.get_tile(0), beta, dst_single.get_tile(0), axis);
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
            Y diff = std::abs(Y(tile_local[i]) - Y(tile2_local[i]));
            Y abs = std::abs(Y(tile_local[i]));
            TEST_ASSERT(diff/abs < 10*T::epsilon());
        }
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>(1.0, 0.0, {11}, {5}, 0);
    check<T>(-1.0, 1.0, {11, 12}, {5, 6}, 0);
    check<T>(2.0, 0.0, {11, 12}, {5, 6}, 1);
    check<T>(1.0, 1.0, {11, 12, 13}, {5, 6, 5}, 0);
    check<T>(-1.0, -1.0, {11, 12, 13}, {5, 6, 5}, 1);
    check<T>(2.0, 0.5, {11, 12, 13}, {5, 6, 5}, 2);
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh4 = {4}, sh33 = {3, 3},
        sh24 = {2, 4}, sh13 = {1, 3}, sh_ = {}, sh22 = {2, 2};
    TensorTraits trA(sh34, sh23), trB(sh23, sh23), trC(sh4, sh4),
        trD(sh33, sh23), trE(sh24, sh13), trF(sh_, sh_), trG(sh24, sh22);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0}, dist00 = {0, 0};
    Tensor<T> A(trA, dist0000, last_tag), B(trB, dist0, last_tag),
        C(trC, dist0, last_tag), D(trD, dist00, last_tag),
        E(trE, dist0000, last_tag), F(trF, dist0, last_tag),
        G(trG, dist00, last_tag);
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, C, 0));
    TEST_THROW(sumprod_fiber<T>(1.0, F, F, 1.0, F, 0));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, B, -1));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, B, 2));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, D, 0));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, E, 0));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, B, 0));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, B, 1));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, G, 0));
    TEST_THROW(sumprod_fiber<T>(1.0, A, A, 1.0, G, 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::sumprod_fiber::init();
    starpu::subcopy::init();
    starpu::clear::init();
    starpu::sumprod_fiber::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::clear::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
