/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/copy.cc
 * Copy operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/copy.hh"
#include "nntile/starpu/copy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape,
        const std::vector<Index> &basetile)
{
    using Y = typename T::repr_t;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    // Traits of source and destination tensors
    TensorTraits src_traits(shape, basetile),
                 dst_traits(shape, basetile);
    // Distributions for source and destination tiles
    Index ntiles = src_traits.grid.nelems;
    std::vector<int> src_distr(ntiles), dst_distr(ntiles);
    for(Index i = 0; i < ntiles; ++i)
    {
        src_distr[i] = (i+1) % mpi_size;
        dst_distr[i] = (i*i+2) % mpi_size;
    }
    // Init source tensor
    Tensor<T> src(src_traits, src_distr, last_tag);
    for(Index i = 0; i < ntiles; ++i)
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
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    // Copy tensor
    copy<T>(src, dst);
    // Check copy
    for(Index i = 0; i < ntiles; ++i)
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
                        == Y(src.index_to_linear(global_index)));
            }
            tile_local.release();
        }
    }
}

template<typename T>
void validate()
{
    check<T>({}, {});
    check<T>({11, 12, 13}, {11, 12, 13});
    check<T>({11, 12, 13}, {3, 4, 5});
    check<T>({1000, 1000}, {450, 450});
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh33 = {3, 3},
        sh24 = {2, 4};
    TensorTraits trA(sh34, sh23), trB(sh33, sh23), trC(sh34, sh24);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist00 = {0, 0};
    Tensor<T> A(trA, dist0000, last_tag),
        B(trB, dist00, last_tag),
        C(trC, dist00, last_tag);
    TEST_THROW(copy<T>(A, B));
    TEST_THROW(copy<T>(A, C));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::copy::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    validate<nntile::int64_t>();
    return 0;
}
