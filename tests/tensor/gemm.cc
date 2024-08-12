/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tensor/gemm.cc
 * GEMM operation on Tensor<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/gemm.hh"
#include "nntile/tile/gemm.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    // Sync to be sure old tags are destroyed on all nodes
    starpu_mpi_barrier(MPI_COMM_WORLD);
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_size = starpu_mpi_world_size();
    int mpi_root = 0;
    starpu_mpi_tag_t last_tag = 0;
    TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    Scalar one = 1, zero = 0;
    // Init single-tiled tensors
    std::vector<Index> sh2222 = {2, 2, 2, 2, 2};
    TensorTraits tr2222(sh2222, sh2222);
    std::vector<int> dist0 = {mpi_root};
    Tensor<T> A_single(tr2222, dist0, last_tag),
        B_single(tr2222, dist0, last_tag),
        C_single(tr2222, dist0, last_tag),
        D_single(tr2222, dist0, last_tag);
    auto A_single_tile = A_single.get_tile(0),
         B_single_tile = B_single.get_tile(0),
         C_single_tile = C_single.get_tile(0),
         D_single_tile = D_single.get_tile(0);
    if(mpi_rank == mpi_root)
    {
        auto A_single_local = A_single_tile.acquire(STARPU_W),
             B_single_local = B_single_tile.acquire(STARPU_W),
             C_single_local = C_single_tile.acquire(STARPU_W),
             D_single_local = D_single_tile.acquire(STARPU_W);
        for(Index i = 0; i < A_single.nelems; ++i)
        {
            A_single_local[i] = Y(i+1);
            B_single_local[i] = 2 * Y(A_single_local[i]);
            C_single_local[i] = 3 * Y(A_single_local[i]);
            D_single_local[i] = C_single_local[i];
        }
        A_single_local.release();
        B_single_local.release();
        C_single_local.release();
        D_single_local.release();
    }
    // Scatter A, B and C
    std::vector<int> distr(32);
    for(Index i = 0; i < distr.size(); ++i)
    {
        distr[i] = (i+1) % mpi_size;
    }
    std::vector<Index> sh1111 = {1, 1, 1, 1, 1};
    TensorTraits tr1111(sh2222, sh1111);
    Tensor<T> A(tr1111, distr, last_tag),
        B(tr1111, distr, last_tag),
        D(tr1111, distr, last_tag);
    scatter<T>(A_single, A);
    scatter<T>(B_single, B);
    scatter<T>(D_single, D);
    // Check default parameters
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opN, B_single_tile, zero,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(one, opN, A, opN, B, zero, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check transA=opT
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opT, A_single_tile, opN, B_single_tile, zero,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(one, opT, A, opN, B, zero, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check transB=opT
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opT, B_single_tile, zero,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(one, opN, A, opT, B, zero, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check transA=transB=opT
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opT, A_single_tile, opT, B_single_tile, zero,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(one, opT, A, opT, B, zero, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check alpha=2
    Scalar two = 2;
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(two, opN, A_single_tile, opN, B_single_tile, zero,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(two, opN, A, opN, B, zero, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check beta=1
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opN, B_single_tile, one,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(one, opN, A, opN, B, one, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check beta=-1
    Scalar mone = -1;
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opN, B_single_tile, mone,
                C_single_tile, 2, 1);
    }
    tensor::gemm<T>(one, opN, A, opN, B, mone, D, 2, 1);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(Y(C_single_local[i]) == Y(D_single_local[i]));
        }
        C_single_local.release();
        D_single_local.release();
    }
}

template<typename T>
void validate()
{
    check<T>();
    // Barrier to wait for cleanup of previously used tags
    starpu_mpi_barrier(MPI_COMM_WORLD);
    // Check throwing exceptions
    starpu_mpi_tag_t last_tag = 0;
    TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    Scalar one = 1, zero = 0;
    std::vector<Index> shape11 = {1, 1}, shape12 = {1, 2},
        shape21 = {2, 1}, shape22 = {2, 2}, shape333 = {3, 3, 3};
    TensorTraits tr11(shape11, shape11), tr12(shape12, shape12),
        tr21(shape21, shape21), tr22(shape22, shape22),
        tr12_(shape12, shape11), tr21_(shape21, shape11),
        tr333(shape333, shape333);
    std::vector<int> dist0 = {0}, dist00 = {0, 0};
    Tensor<T> mat11(tr11, dist0, last_tag),
        mat12(tr12, dist0, last_tag),
        mat12_(tr12_, dist00, last_tag),
        mat21(tr21, dist0, last_tag),
        mat21_(tr21_, dist00, last_tag),
        mat22(tr22, dist0, last_tag),
        mat333(tr333, dist0, last_tag);
    auto fail_trans_val = static_cast<TransOp::Value>(-1);
    auto opF = *reinterpret_cast<TransOp *>(&fail_trans_val);
    // Check ndim
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, -1, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat333, one, mat11, 3, 0));
    TEST_THROW(gemm<T>(one, opT, mat333, opT, mat11, one, mat11, 3, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 2, 0));
    // Check batch_ndim
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 1, 1));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat333, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat333, opT, mat11, one, mat11, 2, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 1, -1));
    // Check incorrect transpositions
    TEST_THROW(gemm<T>(one, opF, mat11, opN, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opF, mat11, opT, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat11, opF, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opF, mat11, one, mat11, 1, 0));
    // Check A and B compatibility
    TEST_THROW(gemm<T>(one, opN, mat12, opN, mat12, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat12, opN, mat21_, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat12, opT, mat21, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat12, opT, mat12_, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat21, opN, mat12, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat21_, opN, mat21, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat21, opT, mat21, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat21_, opT, mat12, one, mat11, 1, 0));
    // Check A and C compatibility
    TEST_THROW(gemm<T>(one, opN, mat21, opN, mat12, one, mat12, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat21_, opN, mat12_, one, mat22, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat12, opN, mat12, one, mat12, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat12_, opN, mat12_, one, mat22, 1, 0));
    // Check B and C compatibility
    TEST_THROW(gemm<T>(one, opN, mat11, opN, mat12, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat11, opN, mat12_, one, mat12, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat11, opT, mat21, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat11, opT, mat21_, one, mat12, 1, 0));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::gemm::init();
    starpu::subcopy::init();
    starpu::gemm::restrict_where(STARPU_CPU);
    starpu::subcopy::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
