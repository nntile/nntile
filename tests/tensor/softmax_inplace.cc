/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/softmax_inplace.cc
 * softmax_inplace operation for Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/softmax_inplace.hh"
#include "nntile/tile/softmax_inplace.hh"
#include "nntile/starpu/softmax_inplace.hh"
#include "nntile/tensor/clear.hh"
#include "nntile/tensor/maxsumexp.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/starpu/clear.hh"
#include "nntile/starpu/maxsumexp.hh"
#include "nntile/starpu/subcopy.hh"
#include "nntile/starpu/copy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check(const std::vector<Index> &shape, const std::vector<Index> &basetile,
        Index axis)
{
    using Y = typename T::repr_t;
    constexpr Scalar alpha = 1.0;
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Some preparation
    starpu_mpi_tag_t last_tag = 0;
    int mpi_size = starpu_mpi_world_size();
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_root = 0;
    // Generate single-tile destination tensor and init it
    TensorTraits dst_single_traits(shape, shape);
    std::vector<int> dist_root = {mpi_root};
    Tensor<T> dst_single(dst_single_traits, dist_root, last_tag);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
        auto tile_local = tile.acquire(STARPU_W);
        for(Index i = 0; i < dst_single.nelems; ++i)
        {
            tile_local[i] = Y(i);
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
    // Generate source tensor as a result of maxsumexp operation
    TensorTraits src_traits(src_shape, src_basetile);
    std::vector<int> src_distr(src_traits.grid.nelems);
    for(Index i = 0; i < src_traits.grid.nelems; ++i)
    {
        src_distr[i] = (i*i+1) % mpi_size;
    }
    Tensor<T> src(src_traits, src_distr, last_tag);
    clear<T>(src);
    maxsumexp<T>(dst, src, axis);
    // Collect source in a single-tile tensor
    TensorTraits src_single_traits(src_shape, src_shape);
    Tensor<T> src_single(src_single_traits, dist_root, last_tag);
    gather<T>(src, src_single);
    // Perform tensor-wise and tile-wise softmax_inplace operations
    softmax_inplace<T>(src, alpha, dst, axis);
    if(mpi_rank == mpi_root)
    {
        tile::softmax_inplace<T>(src_single.get_tile(0), alpha,
                dst_single.get_tile(0), axis);
    }
    // Compare results
    Tensor<T> dst2_single(dst_single_traits, dist_root, last_tag);
    gather<T>(dst, dst2_single);
    if(mpi_rank == mpi_root)
    {
        auto tile = dst_single.get_tile(0);
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
    constexpr Scalar alpha = 1.0;
    check<T>({11}, {5}, 0);
    check<T>({11, 12}, {5, 6}, 0);
    check<T>({11, 12}, {5, 6}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 0);
    check<T>({11, 12, 13}, {5, 6, 5}, 1);
    check<T>({11, 12, 13}, {5, 6, 5}, 2);
    check<T>({1000, 1000}, {450, 450}, 0);
    check<T>({1000, 1000}, {450, 450}, 1);
    // Sync to guarantee old data tags are cleaned up and can be reused
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    std::vector<Index> sh34 = {3, 4}, sh23 = {2, 3}, sh3 = {3}, sh24 = {2, 4},
        sh22 = {2, 2}, sh_ = {}, sh33 = {3, 3}, sh13 = {1, 3}, sh2 = {2};
    TensorTraits trA(sh34, sh23), trB(sh3, sh3), trC(sh24, sh22),
        trD(sh_, sh_), trE(sh33, sh23), trF(sh23, sh13), trG(sh23, sh23);
    std::vector<int> dist0000 = {0, 0, 0, 0}, dist0 = {0}, dist00 = {0, 0};
    Tensor<T> A(trA, dist0000, last_tag), B(trB, dist0, last_tag),
        C(trC, dist00, last_tag), D(trD, dist0, last_tag),
        E(trE, dist00, last_tag), F(trF, dist00, last_tag),
        G(trG, dist0, last_tag);
    TEST_THROW(softmax_inplace<T>(B, alpha, A, 0));
    TEST_THROW(softmax_inplace<T>(D, alpha, D, 0));
    TEST_THROW(softmax_inplace<T>(C, alpha, A, 0));
    TEST_THROW(softmax_inplace<T>(C, alpha, A, -1));
    TEST_THROW(softmax_inplace<T>(C, alpha, A, 2));
    TEST_THROW(softmax_inplace<T>(E, alpha, A, 0));
    TEST_THROW(softmax_inplace<T>(F, alpha, A, 0));
    TEST_THROW(softmax_inplace<T>(C, alpha, A, 1));
    TEST_THROW(softmax_inplace<T>(G, alpha, A, 0));
    TEST_THROW(softmax_inplace<T>(G, alpha, A, 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::softmax_inplace::init();
    starpu::maxsumexp::init();
    starpu::clear::init();
    starpu::subcopy::init();
    starpu::copy::init();
    starpu::softmax_inplace::restrict_where(STARPU_CPU);
    starpu::maxsumexp::restrict_where(STARPU_CPU);
    starpu::clear::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    starpu::copy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
