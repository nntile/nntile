/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/bias.cc
 * Bias operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/tile/bias.hh"
#include "nntile/starpu/bias.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile,
        Index axis)
{
    // Some preparation
    starpu_mpi_tag_t last_tag = 1;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile destination tensor and init it
    Tensor<T> dst_single({shape, shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile_handle = dst_single.get_tile_handle(0);
        auto tile_handle_local = tile_handle.acquire(STARPU_W);
        for(Index i = 0; i < dst_single.nelems; ++i)
        {
            tile_handle_local[i] = T(i);
        }
        tile_handle_local.release();
    }
    // Scatter destination tensor
    TensorTraits dst_traits(shape, basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i+1) % mpi_root;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    scatter<T>(dst_single, dst);
    // Define proper shape and basetile for the source tensor
    std::vector<Index> src_shape(dst_traits.ndim-1),
        src_basetile(dst_traits.ndim-1);
    for(Index i = 0; i < axis; ++i)
    {
        src_shape[i] = dst_traits.shape[i];
        src_basetile[i] = dst_traits.basetile_shape[i];
    }
    for(Index i = axis+1; i < dst_traits.ndim; ++i)
    {
        src_shape[i-1] = dst_traits.shape[i];
        src_basetile[i-1] = dst_traits.basetile_shape[i];
    }
    // Generate single-tile source tensor and init it
    Tensor<T> src_single({src_shape, src_shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile_handle = src_single.get_tile_handle(0);
        auto tile_handle_local = tile_handle.acquire(STARPU_W);
        for(Index i = 0; i < src_single.nelems; ++i)
        {
            tile_handle_local[i] = T(-i);
        }
        tile_handle_local.release();
    }
    // Scatter source tensor
    TensorTraits src_traits(src_shape, src_basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i*i+1) % mpi_root;
    }
    Tensor<T> src(src_traits, src_distr, last_tag);
    scatter<T>(src_single, src);
    // Perform tensor-wise and tile-wise bias operations
    bias<T>(src, dst, axis);
    tile::bias<T>(src_single.get_tile(0), dst_single.get_tile(0), axis);
    // Compare results
    Tensor<T> dst2_single({shape, shape}, {mpi_root}, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto dst_tile_handle = dst_single.get_tile_handle(0);
        auto dst2_tile_handle = dst2_single.get_tile_handle(0);
        auto dst_tile_handle_local = dst_tile_handle.acquire(STARPU_R);
        auto dst2_tile_handle_local = dst2_tile_handle.acquire(STARPU_R);
        for(Index i = 0; i < dst_traits.nelems; ++i)
        {
            TEST_ASSERT(dst_tile_handle_local[i] == dst2_tile_handle_local[i]);
        }
    }
}

template<typename T>
void validate()
{
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::bias::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

