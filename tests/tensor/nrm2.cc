/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/nrm2.cc
 * NRM2 operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/nrm2.hh"
#include "nntile/tile/nrm2.hh"
#include "nntile/tile/clear.hh"
#include "nntile/starpu/nrm2.hh"
#include "nntile/starpu/hypot.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/scal_inplace.hh"
#include "../testing.hh"
#include <limits>

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
    // Create temporary tensor
    std::vector<Index> tmp_basetile(shape);
    for(Index i = 0; i < shape.size(); ++i)
    {
        tmp_basetile[i] = 1;
    }
    TensorTraits tmp_traits(src_traits.grid.shape, tmp_basetile);
    Tensor<T> tmp(tmp_traits, src_distr);
    // Generate output destination tensor
    TensorTraits dst_traits({}, {});
    Tensor<T> dst(dst_traits, dist_root);
    Tensor<T> dst2(dst_traits, dist_root);
    if(mpi_rank == mpi_root)
    {
        Y dst_init = 1.54;
        auto dst_tile = dst.get_tile(0).acquire(STARPU_W);
        auto dst2_tile = dst2.get_tile(0).acquire(STARPU_W);
        dst_tile[0] = dst_init;
        dst2_tile[0] = dst_init;
        dst_tile.release();
        dst2_tile.release();
    }
    // Perform tensor-wise and tile-wise nrm2 operations
    Scalar alpha = -3.1, beta = 0.67;
    nrm2<T>(alpha, src, beta, dst, tmp);
    if(mpi_rank == mpi_root)
    {
        tile::Tile<T> tmp_single({});
        tile::nrm2<T>(alpha, src_single.get_tile(0), beta, dst2.get_tile(0),
                tmp_single);
    }
    // Compare results
    if(mpi_rank == mpi_root)
    {
        auto tile = dst.get_tile(0);
        auto tile2 = dst2.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        Y diff = std::abs(Y(tile_local[0]) - Y(tile2_local[0]));
        Y abs = std::abs(Y(tile_local[0]));
        TEST_ASSERT(diff/abs < 10*T::epsilon());
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({11}, {5});
    check<T>({11, 12}, {5, 6});
    check<T>({11, 12, 13}, {5, 6, 5});
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
}

int main(int argc, char **argv)
{
    int ncpus=1, ncuda=0, cublas=0, ooc=0, ooc_disk_node_id=-1, verbose=0;
    const char *ooc_path = "/tmp/nntile_ooc";
    size_t ooc_size = 16777216;
    starpu::config.init(ncpus, ncuda, cublas, ooc, ooc_path, ooc_size,
        ooc_disk_node_id, verbose);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
