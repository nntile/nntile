/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/gather.cc
 * Gather operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/gather.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "nntile/context.hh"
#include "nntile/starpu/config.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape,
        const std::vector<Index> &src_basetile, int mpi_root)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    mpi_root = mpi_root % mpi_size;
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
    Tensor<T> src(src_traits, src_distr);
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
                tile_local_ptr[j] = Y(src.index_to_linear(global_index));
            }
            tile_local.release();
        }
    }
    // Define destination tensor
    Tensor<T> dst(dst_traits, dst_distr);
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
            TEST_ASSERT(Y(tile_local_ptr[j]) == Y(j));
        }
        tile_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({}, {}, 0);
    check<T>({2, 3, 4}, {2, 3, 4}, 1);
    check<T>({11, 12, 13}, {2, 3, 4}, 0);
    check<T>({1000, 1000}, {450, 450}, 0);
    check<T>({1000, 1000}, {450, 450}, 1);
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    std::vector<Index> sh233 = {2, 3, 3}, sh234 = {2, 3, 4}, sh235 = {2, 3, 5},
        sh23 = {2, 3};
    TensorTraits trA(sh234, sh234), trB(sh235, sh235), trC(sh234, sh233),
        trD(sh23, sh23);
    std::vector<int> dist0 = {0}, dist00 = {0, 0};
    Tensor<T> A(trA, dist0),
        B(trB, dist0),
        C(trC, dist00),
        D(trD, dist0);
    TEST_THROW(gather<T>(A, C));
    TEST_THROW(gather<T>(D, A));
    TEST_THROW(gather<T>(B, A));
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
