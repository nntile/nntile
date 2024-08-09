/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file tests/tile/gemm.cc
 * GEMM operation on Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/tile/gemm.hh"
#include "nntile/starpu/gemm.hh"
#include "../testing.hh"

using namespace nntile;
using namespace nntile::tile;

template<typename T>
void check()
{
    using Y = typename T::repr_t;
    // Check all parameters are properly passed to the underlying gemm
    TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    Scalar one = 1, zero = 0;
    Tile<T> A({2, 2, 2, 2, 2}), B(A.shape), C(A.shape), D(A.shape);
    auto A_local = A.acquire(STARPU_W), B_local = B.acquire(STARPU_W),
         C_local = C.acquire(STARPU_W), D_local = D.acquire(STARPU_W);
    for(Index i = 0; i < A.nelems; ++i)
    {
        A_local[i] = Y(i+1);
        B_local[i] = 2 * Y(A_local[i]);
        C_local[i] = 3 * Y(A_local[i]);
        D_local[i] = C_local[i];
    }
    A_local.release();
    B_local.release();
    C_local.release();
    D_local.release();
    // Check default parameters
    starpu::gemm::submit<T>(opN, opN, 4, 4, 4, 2, one, A, B, zero, C);
    gemm<T>(one, opN, A, opN, B, zero, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
    // Check transA=opT
    starpu::gemm::submit<T>(opT, opN, 4, 4, 4, 2, one, A, B, zero, C);
    gemm<T>(one, opT, A, opN, B, zero, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
    // Check transB=opT
    starpu::gemm::submit<T>(opN, opT, 4, 4, 4, 2, one, A, B, zero, C);
    gemm<T>(one, opN, A, opT, B, zero, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
    // Check transA=transB=opT
    starpu::gemm::submit<T>(opT, opT, 4, 4, 4, 2, one, A, B, zero, C);
    gemm<T>(one, opT, A, opT, B, zero, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
    // Check alpha=2
    Scalar two = 2;
    starpu::gemm::submit<T>(opN, opN, 4, 4, 4, 2, two, A, B, zero, C);
    gemm<T>(two, opN, A, opN, B, zero, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
    // Check beta=1
    starpu::gemm::submit<T>(opN, opN, 4, 4, 4, 2, one, A, B, one, C);
    gemm<T>(one, opN, A, opN, B, one, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
    // Check beta=-1
    Scalar mone = -1;
    starpu::gemm::submit<T>(opN, opN, 4, 4, 4, 2, one, A, B, mone, C);
    gemm<T>(one, opN, A, opN, B, mone, D, 2, 1);
    C_local.acquire(STARPU_R);
    D_local.acquire(STARPU_R);
    for(Index i = 0; i < C.nelems; ++i)
    {
        TEST_ASSERT(Y(C_local[i]) == Y(D_local[i]));
    }
    C_local.release();
    D_local.release();
}

template<typename T>
void validate()
{
    // Check normal execution
    check<T>();
    // Check throwing exceptions
    TransOp opT(TransOp::Trans), opN(TransOp::NoTrans);
    Scalar one = 1, zero = 0;
    Tile<T> mat11({1, 1}), mat12({1, 2}), mat21({2, 1}), mat22({2, 2}),
        mat333({3, 3, 3});
    auto fail_trans_val = static_cast<TransOp::Value>(-1);
    auto opF = *reinterpret_cast<TransOp *>(&fail_trans_val);
    // Check ndim
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, -1, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat333, one, mat11, 3, 0));
    TEST_THROW(gemm<T>(one, opT, mat333, opT, mat11, one, mat11, 3, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 2, 0));
    // Check batch_ndim
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 1, 1));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 1, -1));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat333, one, mat11, 2, 0));
    TEST_THROW(gemm<T>(one, opT, mat333, opT, mat11, one, mat11, 1, 2));
    TEST_THROW(gemm<T>(one, opT, mat11, opT, mat11, one, mat11, 2, -1));
    // Check incorrect transpositions
    TEST_THROW(gemm<T>(one, opF, mat11, opN, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opF, mat11, opT, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat11, opF, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat11, opF, mat11, one, mat11, 1, 0));
    // Check A and B compatibility
    TEST_THROW(gemm<T>(one, opN, mat12, opN, mat12, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat12, opT, mat21, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat21, opN, mat12, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat21, opT, mat21, one, mat11, 1, 0));
    // Check A and C compatibility
    TEST_THROW(gemm<T>(one, opN, mat21, opN, mat11, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opT, mat12, opN, mat11, one, mat11, 1, 0));
    // Check B and C compatibility
    TEST_THROW(gemm<T>(one, opN, mat11, opN, mat12, one, mat11, 1, 0));
    TEST_THROW(gemm<T>(one, opN, mat11, opT, mat21, one, mat11, 1, 0));
}

int main(int argc, char **argv)
{
    // Init StarPU for testing on CPU only
    starpu::Config starpu(1, 0, 0);
    // Init codelet
    starpu::gemm::init();
    starpu::gemm::restrict_where(STARPU_CPU);
    // Launch all tests
    validate<fp32_t>();
    validate<fp64_t>();
    return 0;
}
