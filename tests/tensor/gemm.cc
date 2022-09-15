/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/gemm.cc
 * GEMM operation on Tensor<T>
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2022-09-15
 * */

#include "nntile/tensor/gemm.hh"
#include "nntile/tile/gemm.hh"
#include "nntile/tensor/gather.hh"
#include "nntile/tensor/scatter.hh"
#include "nntile/starpu/gemm.hh"
#include "nntile/starpu/subcopy.hh"
#include "../testing.hh"
#include "../starpu/common.hh"

using namespace nntile;
using namespace nntile::tensor;

template<typename T>
void check()
{
    // Sync to be sure old tags are destroyed on all nodes
    starpu_mpi_barrier(MPI_COMM_WORLD);
    int mpi_rank = starpu_mpi_world_rank();
    int mpi_size = starpu_mpi_world_size();
    int mpi_root = 0;
    starpu_mpi_tag_t last_tag = 0;
    TransOp opT = TransOp::Trans, opN = TransOp::NoTrans;
    T one = 1, zero = 0;
    // Init single-tiled tensors
    Tensor<T> A_single({{2, 2, 2, 2}, {2, 2, 2, 2}}, {mpi_root}, last_tag),
        B_single({{2, 2, 2, 2}, {2, 2, 2, 2}}, {mpi_root}, last_tag),
        C_single({{2, 2, 2, 2}, {2, 2, 2, 2}}, {mpi_root}, last_tag),
        D_single({{2, 2, 2, 2}, {2, 2, 2, 2}}, {mpi_root}, last_tag);
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
            A_single_local[i] = T(i+1);
            B_single_local[i] = 2 * A_single_local[i];
            C_single_local[i] = 3 * A_single_local[i];
            D_single_local[i] = C_single_local[i];
        }
        A_single_local.release();
        B_single_local.release();
        C_single_local.release();
        D_single_local.release();
    }
    // Scatter A, B and C
    std::vector<int> distr(16);
    for(Index i = 0; i < 16; ++i)
    {
        distr[i] = (i+1) % mpi_size;
    }
    Tensor<T> A({{2, 2, 2, 2}, {1, 1, 1, 1}}, distr, last_tag),
        B({{2, 2, 2, 2}, {1, 1, 1, 1}}, distr, last_tag),
        D({{2, 2, 2, 2}, {1, 1, 1, 1}}, distr, last_tag);
    scatter<T>(A_single, A);
    scatter<T>(B_single, B);
    scatter<T>(D_single, D);
    // Check default parameters
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opN, B_single_tile, zero,
                C_single_tile, 2);
    }
    tensor::gemm<T>(one, opN, A, opN, B, zero, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check transA=opT
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opT, A_single_tile, opN, B_single_tile, zero,
                C_single_tile, 2);
    }
    tensor::gemm<T>(one, opT, A, opN, B, zero, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check transB=opT
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opT, B_single_tile, zero,
                C_single_tile, 2);
    }
    tensor::gemm<T>(one, opN, A, opT, B, zero, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check transA=transB=opT
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opT, A_single_tile, opT, B_single_tile, zero,
                C_single_tile, 2);
    }
    tensor::gemm<T>(one, opT, A, opT, B, zero, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check alpha=2
    T two = 2;
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(two, opN, A_single_tile, opN, B_single_tile, zero,
                C_single_tile, 2);
    }
    tensor::gemm<T>(two, opN, A, opN, B, zero, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check beta=1
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opN, B_single_tile, one,
                C_single_tile, 2);
    }
    tensor::gemm<T>(one, opN, A, opN, B, one, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
        }
        C_single_local.release();
        D_single_local.release();
    }
    // Check beta=-1
    T mone = -1;
    if(mpi_rank == mpi_root)
    {
        tile::gemm<T>(one, opN, A_single_tile, opN, B_single_tile, mone,
                C_single_tile, 2);
    }
    tensor::gemm<T>(one, opN, A, opN, B, mone, D, 2);
    gather<T>(D, D_single);
    if(mpi_rank == mpi_root)
    {
        auto C_single_local = C_single_tile.acquire(STARPU_R);
        auto D_single_local = D_single_tile.acquire(STARPU_R);
        for(Index i = 0; i < D.nelems; ++i)
        {
            TEST_ASSERT(C_single_local[i] == D_single_local[i]);
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
    TransOp opT = TransOp::Trans, opN = TransOp::NoTrans;
    T one = 1, zero = 0;
    Tensor<T> mat11({{1, 1}, {1, 1}}, {0}, last_tag),
        mat12({{1, 2}, {1, 2}}, {0}, last_tag),
        mat12_({{1, 2}, {1, 1}}, {0, 0}, last_tag),
        mat21({{2, 1}, {2, 1}}, {0}, last_tag),
        mat21_({{2, 1}, {1, 1}}, {0, 0}, last_tag),
        mat22({{2, 2}, {2, 2}}, {0}, last_tag),
        mat333({{3, 3, 3}, {3, 3, 3}}, {0}, last_tag);
    auto fail_trans_val = static_cast<TransOp::Value>(-1);
    auto opF = *reinterpret_cast<TransOp *>(&fail_trans_val);
    // Check ndim
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, -1));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat333, one, mat11, 3));
    TEST_THROW(gemm<T>(one, opT, mat333, opT, mat11, one, mat11, 3));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 2));
    // Check incorrect transpositions
    TEST_THROW(gemm<T>(one, opF, mat11, opN, mat11, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opF, mat11, opT, mat11, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opN, mat11, opF, mat11, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opT, mat11, opF, mat11, one, mat11, 1));
    // Check A and B compatibility
    TEST_THROW(gemm<T>(one, opN, mat12, opN, mat12, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opN, mat12, opN, mat21_, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opN, mat12, opT, mat21, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opN, mat12, opT, mat12_, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opT, mat21, opN, mat12, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opT, mat21_, opN, mat21, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opT, mat21, opT, mat21, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opT, mat21_, opT, mat12, one, mat11, 1));
    // Check A and C compatibility
    TEST_THROW(gemm<T>(one, opN, mat21, opN, mat12, one, mat12, 1));
    TEST_THROW(gemm<T>(one, opN, mat21_, opN, mat12_, one, mat22, 1));
    TEST_THROW(gemm<T>(one, opT, mat12, opN, mat12, one, mat12, 1));
    TEST_THROW(gemm<T>(one, opT, mat12_, opN, mat12_, one, mat22, 1));
    // Check B and C compatibility
    TEST_THROW(gemm<T>(one, opN, mat11, opN, mat12, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opN, mat11, opN, mat12_, one, mat12, 1));
    TEST_THROW(gemm<T>(one, opN, mat11, opT, mat21, one, mat11, 1));
    TEST_THROW(gemm<T>(one, opN, mat11, opT, mat21_, one, mat12, 1));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing
    StarpuTest starpu;
    // Init codelet
    starpu::gemm::init();
    starpu::subcopy::init();
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}

