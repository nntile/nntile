/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/normalize.cc
 * normalize operation for Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-15
 * */

#include "nntile/tensor/normalize.hh"
#include "nntile/tile/normalize.hh"
#include "nntile/starpu/normalize.hh"
#include "nntile/tensor/sumnorm.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/sumnorm.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile,
        Index axis)
{
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile destination tensor and init it
    Tensor<T> dst_single({shape, shape}, {mpi_root}, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_single.nelems; ++i)
        {
            tile_local[i] = T(i);
        }
        tile_local.release();
    }
    // Scatter destination tensor
    TensorTraits dst_traits(shape, basetile);
    std::vector<int> dst_distr(dst_traits.grid.nelems);
    for(Index i = 0; i < dst_traits.grid.nelems; ++i)
    {
        dst_distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> dst(dst_traits, dst_distr, last_tag);
    scatter<T>(dst_single, dst);
    // Define proper shape and basetile for the source tensor
    std::vector<Index> src_shape(dst_traits.ndim),
        src_basetile(dst_traits.ndim);
    src_shape[0] = 2;
    src_basetile[0] = 2;
    for(Index i = 1; i <= axis; ++i)
    {
        src_shape[i] = shape[i-1];
        src_basetile[i] = basetile[i-1];
    }
    for(Index i = axis+1; i < dst_traits.ndim; ++i)
    {
        src_shape[i] = shape[i];
        src_basetile[i] = basetile[i];
    }
    // Generate source tensor as a result of sumnorm operation
    TensorTraits src_traits(src_shape, src_basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr, last_tag);
    sumnorm<T>(dst, src, axis);
    // Collect source in a single-tile tensor
    Tensor<T> src_single({src_shape, src_shape}, {mpi_root}, last_tag);
    gather<T>(src, src_single);
    // Prepare all other parameters
    Tensor<T> gamma_beta({{2}, {2}}, {mpi_root}, last_tag);
    {
        auto gb_tile = gamma_beta.get_tile(0);
        auto gb_local = gb_tile.acquire(STARPU_W);
        gb_local[0] = T{1};
        gb_local[1] = T{0};
        gb_local.release();
    }
    // Perform tensor-wise and tile-wise normalize operations
    normalize<T>(gamma_beta.get_tile(0), src, dst, shape[axis], T{0}, axis);
    if(mpi_rank == mpi_root)
    {
        tile::normalize<T>(gamma_beta.get_tile(0), src_single.get_tile(0),
                dst_single.get_tile(0), shape[axis], T{0}, axis);
    }
    // Compare results
    Tensor<T> dst2_single({shape, shape}, {mpi_root}, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile2 = dst2_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_R);
        auto tile2_local = tile2.acquire(STARPU_R);
        for(Index i = 0; i < dst_traits.nelems; ++i)
        {
            TEST_ASSERT(tile_local[i] == tile2_local[i]);
        }
        tile_local.release();
        tile2_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>({11}, {5}, 0);
    check<T>({11, 12}, {5, 6}, 0);
    check<T>({11, 12}, {5, 6}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 0);
    check<T>({11, 12, 13}, {5, 6, 5}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 2);
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    Tensor<T> A({{3, 4}, {2, 3}}, {0, 0, 0, 0}, last_tag),
        B({{3}, {3}}, {0}, last_tag),
        C({{2, 4}, {2, 2}}, {0, 0}, last_tag),
        D({{}, {}}, {0}, last_tag),
        E({{3, 3}, {2, 3}}, {0, 0}, last_tag),
        F({{2, 3}, {1, 3}}, {0, 0}, last_tag),
        G({{2, 3}, {2, 3}}, {0}, last_tag);
    StarpuVariableHandle gamma_beta(2*sizeof(T), STARPU_SCRATCH);
    TEST_THROW(normalize<T>(gamma_beta, B, A, 1, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, D, D, 1, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, C, A, 0, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, C, A, 1, T{-0.1}, 0));
    TEST_THROW(normalize<T>(gamma_beta, C, A, 1, T{0}, -1));
    TEST_THROW(normalize<T>(gamma_beta, C, A, 1, T{0}, 2));
    TEST_THROW(normalize<T>(gamma_beta, E, A, 1, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, F, A, 1, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, C, A, 1, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, C, A, 1, T{0}, 1));
    TEST_THROW(normalize<T>(gamma_beta, G, A, 1, T{0}, 0));
    TEST_THROW(normalize<T>(gamma_beta, G, A, 1, T{0}, 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::normalize::init();
    starpu::sumnorm::init();
    starpu::clear::init();
    starpu::subcopy::init();
    // Restrict execution to CPU to properly compare results
    starpu::normalize::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

