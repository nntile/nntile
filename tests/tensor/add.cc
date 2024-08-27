/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/add.cc
 * Per-element addition of tensors
 *
 * @version 1.0.0
 * */

#include <iostream>
#include "nntile/tensor/add.hh"
#include "nntile/tile/add.hh"
#include "nntile/starpu/add.hh"
#include "nntile/starpu/scal.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
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
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    scatter<T>(dst_single, dst);
    // Define proper shape and basetile for the source tensor

    std::vector<Index> src1_shape(shape), src1_basetile(basetile);
    std::vector<Index> src2_shape(shape), src2_basetile(basetile);

    // Generate single-tile source tensor and init it

    TensorTraits src1_single_traits(src1_shape, src1_shape);
    Tensor<T> src1_single(src1_single_traits, dist_root, last_tag);

    TensorTraits src2_single_traits(src2_shape, src2_shape);
    Tensor<T> src2_single(src2_single_traits, dist_root, last_tag);


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
            tile2_local[i] = Y(-i);
        }
        tile2_local.release();
    }

    // Scatter source tensor
    TensorTraits src1_traits(src1_shape, src1_basetile);
    std::vector<int> src1_distr(src1_traits.grid.nelems);
    TensorTraits src2_traits(src2_shape, src2_basetile);
    std::vector<int> src2_distr(src2_traits.grid.nelems);

    for(Index i = 0; i < src1_traits.grid.nelems; ++i)
    {
        src1_distr[i] = (i*i+1) % mpi_size;
        src2_distr[i] = (i*i+1) % mpi_size;
    }
    
    Tensor<T> src1(src1_traits, src1_distr, last_tag);
    Tensor<T> src2(src2_traits, src2_distr, last_tag);
    scatter<T>(src1_single, src1);
    scatter<T>(src2_single, src2);
    // Perform tensor-wise and tile-wise add operations

    add<T>(-1.0, src1, src2, 0.5, dst);
    if(mpi_rank == mpi_root)
    {
        tile::add<T>(-1.0, src1_single.get_tile(0), src2_single.get_tile(0),  0.5, dst_single.get_tile(0));
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
            TEST_ASSERT(Y(tile_local[i]) == Y(tile2_local[i]));
        }
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    
    check<T>({11}, {5});
    check<T>({11, 12}, {5, 6});
    check<T>({11, 12}, {5, 6});
    check<T>({11, 12, 13}, {5, 6, 5});
    check<T>({11, 12, 13}, {5, 6, 5});
    check<T>({11, 12, 13}, {5, 6, 5});
    check<T>({1000, 1000}, {450, 450});
    check<T>({1000, 1000}, {450, 450});

    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh3 = {3}, sh4 = {4};
    TensorTraits trA(sh34, sh23), trB(sh3, sh3), trC(sh4, sh4);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0};
    Tensor<T> A(trA, dist0000, last_tag), B(trB, dist0, last_tag),  C(trC, dist0, last_tag);

    TEST_THROW(add<T>(1.0, A, B, 0.0, C));
    TEST_THROW(add<T>(1.0, A, C, 0.0, B));
    TEST_THROW(add<T>(1.0, B, C, 0.0, A));
    TEST_THROW(add<T>(1.0, B, A, 0.0, C));
    TEST_THROW(add<T>(1.0, C, B, 0.0, A));
    TEST_THROW(add<T>(1.0, C, A, 0.0, B));
    TEST_THROW(add<T>(1.0, A, A, 0.0, B));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::scal::init();
    starpu::add::init();
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::add::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
