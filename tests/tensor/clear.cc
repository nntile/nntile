/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/clear.cc
 * Clear operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/clear.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/tensor/gather.hh"
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
    // Traits
    TensorTraits src_traits(shape, basetile), dst_traits(shape, shape);
    // Distribution
    Index src_ntiles = src_traits.grid.nelems;
    std::vector<int> src_distr(src_ntiles), dst_distr{0};
    for(Index i = 0; i < src_ntiles; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
    }
    // Init source tensor
    Tensor<T> src(src_traits, src_distr);
    for(Index i = 0; i < src_ntiles; ++i)
    {
        if(src_distr[i] == mpi_rank)
        {
            auto tile_handle = src.get_tile_handle(i);
            auto tile_local = tile_handle.acquire(STARPU_W);
            T *tile_local_ptr = reinterpret_cast<T *>(tile_local.get_ptr());
            auto tile_traits = src.get_tile_traits(i);
            for(Index j = 0; j < tile_traits.nelems; ++j)
            {
                tile_local_ptr[j] = Y(-1);
            }
            tile_local.release();
        }
    }
    // Define destination tensor
    Tensor<T> dst(dst_traits, dst_distr);
    // Clear source and gather into destination
    clear<T>(src);
    gather<T>(src, dst);
    // Check
    if(mpi_rank == dst_distr[0])
    {
        auto tile_handle = dst.get_tile_handle(0);
        auto tile_local = tile_handle.acquire(STARPU_R);
        T *tile_local_ptr = reinterpret_cast<T *>(tile_local.get_ptr());
        for(Index i = 0; i < dst.nelems; ++i)
        {
            TEST_ASSERT(Y(tile_local_ptr[i]) == Y(0));
        }
        tile_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({}, {});
    check<T>({11, 12, 13}, {2, 3, 4});
    check<T>({1000, 1000}, {450, 450});
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
