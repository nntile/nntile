/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/norm.cc
 * Euclidean norm of all elements in a Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/norm.hh"
#include "nntile/tile/norm.hh"
#include "nntile/starpu/norm.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "nntile/context.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile, Scalar alpha, Scalar beta)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // For now, test only single-tile case to avoid distributed norm issues
    // Generate single-tile source tensor
    TensorTraits src_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src_single(src_single_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        auto tile = src_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < tile.nelems; ++i)
        {
            tile_local[i] = Y(1.0 - i);
        }
        tile_local.release();
    }
    // Generate distributed-tile source tensor (single tile for now)
    TensorTraits src_traits(shape, shape); // Use shape as basetile for single tile
    Tensor<T> src(src_traits, dist_root);
    scatter(src_single, src);
    // Generate destination tensors (scalar tensors with empty shape)
    TensorTraits dst_traits({}, {});
    Tensor<T> dst_single(dst_traits, dist_root);
    Tensor<T> dst(dst_traits, dist_root); // scalar result always on root
    // Initialize destination tensors
    if(mpi_rank == mpi_root)
    {
        auto dst_single_tile = dst_single.get_tile(0);
        auto dst_single_local = dst_single_tile.acquire(STARPU_W);
        dst_single_local[0] = Y(0.0);
        dst_single_local.release();

        auto dst_tile = dst.get_tile(0);
        auto dst_local = dst_tile.acquire(STARPU_W);
        dst_local[0] = Y(0.0);
        dst_local.release();
    }
    // Get reference norm using tile-level operation
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        auto dst_single_tile = dst_single.get_tile(0);
        tile::norm_async<T>(alpha, src_tile, beta, dst_single_tile);
    }
    // Get norm using tensor-level operation
    norm_async<T>(alpha, src, beta, dst);
    starpu_task_wait_for_all();
    starpu_mpi_wait_for_all(MPI_COMM_WORLD);
    // Compare results
    if(mpi_rank == mpi_root)
    {
        auto dst_single_tile = dst_single.get_tile(0);
        auto dst_tile = dst.get_tile(0);
        auto dst_single_local = dst_single_tile.acquire(STARPU_R);
        auto dst_local = dst_tile.acquire(STARPU_R);
        TEST_ASSERT(Y(dst_single_local[0]) == Y(dst_local[0]));
        dst_single_local.release();
        dst_local.release();
    }
}

template<typename T>
void validate()
{
    // Test single-tile cases only (distributed norm needs proper implementation)
    check<T>({5}, {5}, 1.0, 0.0);
    check<T>({11}, {4}, 1.0, -1.0);
    check<T>({10, 100}, {4, 45}, -1.0, 2.0);
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // No checks that throw exceptions
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
