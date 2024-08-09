/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/randn.cc
 * Randn operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/randn.hh"
#include "nntile/tile/randn.hh"
#include "nntile/starpu/randn.hh"
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
    unsigned long long seed = -1;
    Scalar mean = -1;
    Scalar stddev = 2;
    std::vector<Index> start(shape.size());
    // Generate single-tile destination tensor
    TensorTraits dst_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> dst_single(dst_single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        tile::randn<T>(dst_single.get_tile(0), start, shape, seed, mean,
                stddev);
    }
    // Generate distributed-tile destination tensor
    TensorTraits dst_traits(shape, basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    randn<T>(dst, start, shape, seed, mean, stddev);
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
    check<T>({}, {});
    check<T>({5}, {5});
    check<T>({11}, {5});
    check<T>({11, 12, 13}, {5, 6, 7});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    unsigned long long seed = -1;
    Scalar mean = 1, stddev = 2;
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh34 = {3, 4}, sh23 = {2,3};
    std::vector<int> dist0000 = {0, 0, 0, 0};
    TensorTraits trA(sh34, sh23);
    Tensor<T> A(trA, dist0000, last_tag);
    TEST_THROW(randn<T>(A, {0}, {3, 4}, seed, mean, stddev));
    TEST_THROW(randn<T>(A, {0, 0}, {3}, seed, mean, stddev));
    TEST_THROW(randn<T>(A, {0, -1}, {3, 4}, seed, mean, stddev));
    TEST_THROW(randn<T>(A, {0, 1}, {3, 4}, seed, mean, stddev));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::randn::init();
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::randn::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
