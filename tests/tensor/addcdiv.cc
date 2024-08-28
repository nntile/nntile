/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/addcdiv.cc
 * Addcdiv operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/addcdiv.hh"
#include "nntile/tile/addcdiv.hh"
#include "nntile/starpu/addcdiv.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(Scalar val, Scalar eps, const std::vector<Index> &shape,
        const std::vector<Index> &basetile)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile source, nom and denom tensors
    TensorTraits single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> src_single(single_traits, dist_root, last_tag),
              nom_single(single_traits, dist_root, last_tag),
              denom_single(single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        // Get the only tiles of single-tiled tensors
        auto src_tile = src_single.get_tile(0);
        auto nom_tile = nom_single.get_tile(0);
        auto denom_tile = denom_single.get_tile(0);
        // Ask StarPU to allocate buffers local to write into them
        auto src_local = src_tile.acquire(STARPU_W);
        auto nom_local = nom_tile.acquire(STARPU_W);
        auto denom_local = denom_tile.acquire(STARPU_W);
        // Init tiles
        for(Index i = 0; i < src_tile.nelems; ++i)
        {
            src_local[i] = Y(i);
            nom_local[i] = Y(i-100);
            denom_local[i] = Y(i+1);
        }
        // Put data back into StarPU
        src_local.release();
        nom_local.release();
        denom_local.release();
    }
    // Generate distributed-tile source and destination tensors
    TensorTraits traits(shape, basetile);
    std::vector<int> src_distr(traits.grid.nelems),
                     nom_distr(src_distr),
                     denom_distr(src_distr);
    for(Index i = 0; i < traits.grid.nelems; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
        nom_distr[i] = (i+2) % mpi_size;
        denom_distr[i] = (i+3) % mpi_size;
    }
    Tensor<T> src(traits, src_distr, last_tag),
              nom(traits, nom_distr, last_tag),
              denom(traits, denom_distr, last_tag);
    scatter(src_single, src);
    scatter(nom_single, nom);
    scatter(denom_single, denom);
    // Get addcdiv
    if(mpi_rank == mpi_root)
    {
        auto src_tile = src_single.get_tile(0);
        auto nom_tile = nom_single.get_tile(0);
        auto denom_tile = denom_single.get_tile(0);
        // Per-tile routine
        tile::addcdiv<T>(val, eps, nom_tile, denom_tile, src_tile);
    }
    addcdiv<T>(val, eps, nom, denom, src);
    // Compare results
    Tensor<T> src2_single(single_traits, dist_root, last_tag);
    gather<T>(src, src2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = src_single.get_tile(0);
        auto tile2 = src2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < traits.nelems; ++i)
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
    check<T>(-10, 1, {5}, {5});
    // check<T>(5, 1e-5, {11}, {5});
    // check<T>(0.2, 1e-2, {11, 12, 13}, {5, 6, 7});
    // check<T>(0.2, 1e-2, {1000, 1000}, {450, 450});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // No checks that throw exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::addcdiv::init();
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::addcdiv::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
