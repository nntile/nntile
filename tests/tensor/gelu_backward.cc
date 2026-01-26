/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/gelu_backward.cc
 * Backward GeLU operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/gelu_backward.hh"
#include "nntile/tile/gelu_backward.hh"
#include "nntile/starpu/gelu_backward.hh"
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
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source tensors
    TensorTraits x_single_traits(shape, shape);
    TensorTraits dy_single_traits(shape, shape);
    TensorTraits dx_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> x_single(x_single_traits, dist_root);
    Tensor<T> dy_single(dy_single_traits, dist_root);
    Tensor<T> dx_single(dx_single_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        auto x_tile = x_single.get_tile(0);
        auto x_tile_local = x_tile.acquire(STARPU_W);
        auto dy_tile = dy_single.get_tile(0);
        auto dy_tile_local = dy_tile.acquire(STARPU_W);
        auto dx_tile = dx_single.get_tile(0);
        auto dx_tile_local = dx_tile.acquire(STARPU_W);
        for(Index i = 0; i < x_tile.nelems; ++i)
        {
            x_tile_local[i] = Y(i % 10 - 5);  // Values from -5 to 4
            dy_tile_local[i] = Y(i % 7 + 1);  // Values from 1 to 7
            dx_tile_local[i] = Y(0.0);        // Initialize to zero
        }
        x_tile_local.release();
        dy_tile_local.release();
        dx_tile_local.release();
    }
    // Generate distributed-tile source tensors
    TensorTraits x_traits(shape, basetile);
    std::vector<int> x_distr(x_traits.grid.nelems);
    for(Index i = 0; i < x_traits.grid.nelems; ++i)
    {
        x_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> x(x_traits, x_distr);
    scatter(x_single, x);

    TensorTraits dy_traits(shape, basetile);
    std::vector<int> dy_distr(dy_traits.grid.nelems);
    for(Index i = 0; i < dy_traits.grid.nelems; ++i)
    {
        dy_distr[i] = (i+2) % mpi_size;
    }
    Tensor<T> dy(dy_traits, dy_distr);
    scatter(dy_single, dy);

    TensorTraits dx_traits(shape, basetile);
    std::vector<int> dx_distr(dx_traits.grid.nelems);
    for(Index i = 0; i < dx_traits.grid.nelems; ++i)
    {
        dx_distr[i] = (i+3) % mpi_size;
    }
    Tensor<T> dx(dx_traits, dx_distr);
    scatter(dx_single, dx);

    // Get GeLU backward
    if(mpi_rank == mpi_root)
    {
        auto x_tile = x_single.get_tile(0);
        auto dy_tile = dy_single.get_tile(0);
        auto dx_tile = dx_single.get_tile(0);
        tile::gelu_backward<T>(x_tile, dy_tile, dx_tile);
    }
    gelu_backward<T>(x, dy, dx);
    // Compare results
    Tensor<T> dx2_single(dx_single_traits, dist_root);
    gather<T>(dx, dx2_single);
    if(mpi_rank == mpi_root)
    {
        auto dx_tile = dx_single.get_tile(0);
        auto dx2_tile = dx2_single.get_tile(0);
        auto dx_tile_local = dx_tile.acquire(STARPU_R);
        auto dx2_tile_local = dx2_tile.acquire(STARPU_R);
        for(Index i = 0; i < dx_traits.nelems; ++i)
        {
            TEST_ASSERT(Y(dx_tile_local[i]) == Y(dx2_tile_local[i]));
        }
        dx_tile_local.release();
        dx2_tile_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({}, {});
    check<T>({5}, {5});
    check<T>({11}, {5});
    check<T>({11, 12, 13}, {5, 6, 7});
    check<T>({1000, 1000}, {450, 450});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // No checks that throw exceptions
}

int main(int argc, char **argv)
{
    // Initialize StarPU and MPI
    int ncpu=1, ncuda=0, ooc=0, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    auto context = Context(ncpu, ncuda, ooc, ooc_path, ooc_size, verbose);

    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();

    return 0;
}
