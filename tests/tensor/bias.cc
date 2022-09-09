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
 * @date 2022-09-08
 * */

#include "nntile/tensor/bias.hh"
#include "nntile/starpu/bias.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const TensorTraits &dst_traits, Index axis)
{
    // Get traits for the source tensor
    std::vector<Index> src_shape(dst_traits.ndim-1),
        src_basetile(dst_traits.ndim-1);
    for(Index i = 0; i < axis; ++i)
    {
        src_shape[i] = dst_traits.shape[i];
        src_basetile[i] = dst_traits.basetile[i];
    }
    for(Index i = axis+1; i < dst_traits.ndim; ++i)
    {
        src_shape[i-1] = dst_traits.shape[i];
        src_basetile[i-1] = dst_traits.basetile[i];
    }
    TensorTraits src_traits(src_shape, src_basetile);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Define destination for the bias operation
    Index dst_ntiles = dst_traits.grid.nelems;
    std::vector<int> dst_distr(dst_ntiles);
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    // Init destination on 0-rank and spread it
    if(mpi_rank == 0)
    {
        Tile<T> dst_as_tile(dst.shape);
        auto dst_as_tile_local = dst_as_tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_as_tile.nelems; ++i)
        {
            dst_as_tile_local[i] = T(i+2);
        }
        dst_as_tile_local.release();
        for(Index i = 0; i < dst_ntiles; ++i)
        {
            auto dst_tile_handle = dst.get_tile_handle(i);
            int dst_tile_rank = starpu_mpi_data_get_rank(dst_tile_handle);
            // Local copying
            if(dst_tile_rank == 0)
            {
            }
            // Copy across MPI
            else
            {
                starpu_mpi_send(dst_tile_handle, dst_tile_rank, 0,
                        MPI_COMM_WORLD);
            }
        }
    }
    else
    {
    }
    // Define several sources of the bias operation
    Index src_ntiles = src_traits.grid.nelems;
    std::vector<int> src_distr(src_ntiles);
    for(Index i = 0; i < src_ntiles; ++i)
    {
        src_distr[i] = (i+2) % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr, last_tag);
    // Define corresponding tiles, that are stored
    Tile<T> src_tile(src.shape), dst_tile(dst.shape), tmp_tile(dst.shape);
    starpu_mpi_data_register(src_tile, 0, last_tag);
    starpu_mpi_data_register(dst_tile, 0, last_tag);
    starpu_mpi_data_register(tmp_tile, 0, last_tag);
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

