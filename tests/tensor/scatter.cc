/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/scatter.cc
 * Scatter operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape,
        const std::vector<Index> &dst_basetile)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Traits of source and destination tensors
    TensorTraits src_traits(shape, shape),
                 dst_traits(shape, dst_basetile);
    // Distributions for source and destination tiles
    Index dst_ntiles = dst_traits.grid.nelems;
    std::vector<int> src_distr{0}, dst_distr(dst_ntiles);
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    // Init source tensor
    Tensor<T> src(src_traits, src_distr, last_tag);
    if(mpi_rank == 0)
    {
        auto tile_handle = src.get_tile_handle(0);
        auto tile_local = tile_handle.acquire(STARPU_W);
        T *tile_local_ptr = reinterpret_cast<T *>(tile_local.get_ptr());
        auto tile_traits = src.get_tile_traits(0);
        for(Index j = 0; j < tile_traits.nelems; ++j)
        {
            tile_local_ptr[j] = Y(j);
        }
        tile_local.release();
    }
    // Define destination tensor
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    // Scatter
    scatter<T>(src, dst);
    // Check scatter
    for(Index i = 0; i < dst_ntiles; ++i)
    {
        if(dst_distr[i] == mpi_rank)
        {
            auto tile_handle = dst.get_tile_handle(i);
            auto tile_local = tile_handle.acquire(STARPU_R);
            T *tile_local_ptr = reinterpret_cast<T *>(tile_local.get_ptr());
            auto tile_traits = dst.get_tile_traits(i);
            auto tile_index = dst.grid.linear_to_index(i);
            for(Index j = 0; j < dst.ndim; ++j)
            {
                tile_index[j] *= dst.basetile_shape[j];
            }
            for(Index j = 0; j < tile_traits.nelems; ++j)
            {
                auto global_index = tile_traits.linear_to_index(j);
                for(Index k = 0; k < dst.ndim; ++k)
                {
                    global_index[k] += tile_index[k];
                }
                TEST_ASSERT(Y(tile_local_ptr[j])
                        == Y(dst.index_to_linear(global_index)));
            }
            tile_local.release();
        }
    }
}

template<typename T>
void validate()
{
    check<T>({}, {});
    check<T>({2, 3, 4}, {2, 3, 4});
    check<T>({11, 12, 13}, {2, 3, 4});
    check<T>({768, 1000}, {384, 500});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh234 = {2, 3, 4}, sh235 = {2, 3 ,5}, sh233 = {2, 3, 3},
        sh23 = {2, 3};
    std::vector<int> dist0 = {0}, dist00 = {0, 0};
    TensorTraits trA(sh234, sh234), trB(sh235, sh235), trC(sh234, sh233),
        trD(sh23, sh23);
    Tensor<T> A(trA, dist0, last_tag),
        B(trB, dist0, last_tag),
        C(trC, dist00, last_tag),
        D(trD, dist0, last_tag);
    TEST_THROW(scatter<T>(C, A));
    TEST_THROW(scatter<T>(A, D));
    TEST_THROW(scatter<T>(A, B));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
