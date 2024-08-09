/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/mask_scalar.cc
 * Mask scalar operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/mask_scalar.hh"
#include "nntile/tile/mask_scalar.hh"
#include "nntile/starpu/mask_scalar.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
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
    Scalar val = -0.5;
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile destination tensor
    TensorTraits data_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> data_single(data_single_traits, dist_root, last_tag);
    std::vector<Index> mask_shape{shape[0], shape[1]};
    TensorTraits mask_single_traits(mask_shape, mask_shape);
    Tensor<bool_t> mask_single(mask_single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = data_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < tile.nelems; ++i)
        {
            tile_local[i] = Y(i);
        }
        tile_local.release();
        auto tile_mask = mask_single.get_tile(0);
        auto tile_mask_local = tile_mask.acquire(STARPU_W);
        for(Index i = 0; i < shape[0]; ++i)
        {
            for(Index j = 0; j < shape[1]; ++j)
            {
                if(i+j % 2 == 0)
                {
                    tile_mask_local[j*shape[0]+i] = bool_t(false);
                }
                else
                {
                    tile_mask_local[j*shape[0]+i] = bool_t(true);
                }
            }
        }
        tile_mask_local.release();
    }
    if(mpi_rank == mpi_root)
    {
        auto data_tile = data_single.get_tile(0);
        auto mask_tile = mask_single.get_tile(0);
        tile::mask_scalar<T>(mask_tile, val, data_tile);
    }
    // Generate distributed-tile destination tensor
    TensorTraits dst_traits(shape, basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    std::vector<Index> mask_basetile{basetile[0], basetile[1]};
    TensorTraits mask_traits(mask_shape, mask_basetile);
    std::vector<int> mask_distr(mask_traits.grid.nelems);
    for(Index i = 0; i < mask_traits.grid.nelems; ++i)
    {
        mask_distr[i] = (i+1) % mpi_size;
    }
    Tensor<bool_t> mask(mask_traits, mask_distr, last_tag);
    scatter<T>(data_single, dst);
    scatter<bool_t>(mask_single, mask);
    std::cout << "Scatter mask is done" << std::endl;
    // Get result of mask scalar operation
    mask_scalar<T>(mask, val, dst, 1);
    std::cout << "Mask scalar is done" << std::endl;
    // Compare results
    Tensor<T> dst2_single(data_single_traits, dist_root, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = data_single.get_tile(0);
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
    // check<T>({}, {});
    check<T>({5, 5, 10}, {5, 5, 10});
    check<T>({10, 10, 4}, {2, 2, 2});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // No checks that throw exceptions
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::mask_scalar::init();
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::mask_scalar::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
