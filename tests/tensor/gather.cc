/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/gather.cc
 * Gather operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-12
 * */

#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape,
        const std::vector<Index> &src_basetile, int mpi_root)
{
    // Some preparation
    starpu_mpi_tag_t last_tag = 1;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Traits of source and destination tensors
    TensorTraits src_traits(shape, src_basetile),
                 dst_traits(shape, shape);
    // Distributions for source and destination tiles
    Index src_ntiles = src_traits.grid.nelems;
    std::vector<int> src_distr(src_ntiles), dst_distr = {mpi_root};
    for(Index i = 0; i < src_ntiles; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
    }
    // Init source tensor
    Tensor<T> src(src_traits, src_distr, last_tag);
    for(Index i = 0; i < src_ntiles; ++i)
    {
        if(src_distr[i] == mpi_rank)
        {
            auto tile_handle = src.get_tile_handle(i);
            auto tile_local = tile_handle.acquire(STARPU_W);
            T *tile_local_ptr = reinterpret_cast<T *>(tile_local.get_ptr());
            auto tile_traits = src.get_tile_traits(i);
            auto tile_index = src.grid.linear_to_index(i);
            for(Index j = 0; j < src.ndim; ++j)
            {
                tile_index[j] *= src.basetile_shape[j];
            }
            for(Index j = 0; j < tile_traits.nelems; ++j)
            {
                auto global_index = tile_traits.linear_to_index(j);
                for(Index k = 0; k < src.ndim; ++k)
                {
                    global_index[k] += tile_index[k];
                }
                tile_local_ptr[j] = T(src.index_to_linear(global_index));
            }
            tile_local.release();
        }
    }
    // Define destination tensor
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    // Gather
    gather<T>(src, dst);
    // Check gather
    if(mpi_rank == mpi_root)
    {
        auto tile_handle = dst.get_tile_handle(0);
        auto tile_local = tile_handle.acquire(STARPU_R);
        T *tile_local_ptr = reinterpret_cast<T *>(tile_local.get_ptr());
        auto tile_traits = dst.get_tile_traits(0);
        for(Index j = 0; j < tile_traits.nelems; ++j)
        {
            TEST_ASSERT(tile_local_ptr[j] == T(j));
        }
        tile_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({}, {}, 0);
    starpu_mpi_barrier(MPI_COMM_WORLD);
    check<T>({2, 3, 4}, {2, 3, 4}, 1);
    starpu_mpi_barrier(MPI_COMM_WORLD);
    check<T>({11, 12, 13}, {2, 3, 4}, 0);
    starpu_mpi_barrier(MPI_COMM_WORLD);
    starpu_mpi_tag_t last_tag = 1;
    Tensor<T> A({{2, 3, 4}, {2, 3, 4}}, {0}, last_tag),
        B({{2, 3, 5}, {2, 3, 5}}, {0}, last_tag),
        C({{2, 3, 4}, {2, 3, 3}}, {0, 0}, last_tag),
        D({{2, 3}, {2, 3}}, {0}, last_tag);
    TEST_THROW(gather<T>(A, C));
    TEST_THROW(gather<T>(D, A));
    TEST_THROW(gather<T>(B, A));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::subcopy::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

