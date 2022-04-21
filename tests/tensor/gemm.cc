#include "nntile/tensor/gemm.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "../testing.hh"
#include <limits>
#include <cmath>
#include <iostream>

using namespace nntile;

template<typename T>
void gemm_naive(TransOp transA, TransOp transB, Index M, Index N, Index K,
        T alpha, const T *A, Index ldA, const T *B, Index ldB, T beta,
        T *C, Index ldC)
{
    for(Index m = 0; m < M; ++m)
    {
        for(Index n = 0; n < N; ++n)
        {
            T res = 0;
            for(Index k = 0; k < K; ++k)
            {
                Index A_offset, B_offset;
                switch(transA.value)
                {
                    case TransOp::NoTrans:
                        A_offset = k*ldA + m;
                        break;
                    default:
                        A_offset = m*ldA + k;
                }
                switch(transB.value)
                {
                    case TransOp::NoTrans:
                        B_offset = n*ldB + k;
                        break;
                    default:
                        B_offset = k*ldB + n;
                }
                res += A[A_offset] * B[B_offset];
            }
            Index C_offset = n*ldC + m;
            if(beta == 0)
            {
                C[C_offset] = alpha * res;
            }
            else
            {
                C[C_offset] = beta*C[C_offset] + alpha*res;
            }
        }
    }
}

template<typename T>
T norm(Index M, Index N, const T *C, Index ldC)
{
    T scale = 1, ssq = 0;
    for(Index m = 0; m < M; ++m)
    {
        for(Index n = 0; n < N; ++n)
        {
            Index C_offset = n*ldC + m;
            T val = C[C_offset];
            if(val > scale)
            {
                T tmp = scale / val;
                scale = val;
                ssq *= tmp * tmp;
                ssq += 1;
            }
            else
            {
                T tmp = val / scale;
                ssq += tmp * tmp;
            }
        }
    }
    return std::sqrt(ssq) * scale;
}

template<typename T>
void test_gemm(TransOp transA, TransOp transB, Index M, Index N, Index K,
        T alpha, const Tensor<T> &A, Index ldA, const Tensor<T> &B, Index ldB,
        T beta, const Tensor<T> &C, Index ldC, const Tensor<T> &D, Index ldD)
{
    Tensor<T> A_local(A.shape, A.shape), B_local(B.shape, B.shape),
        C_local(C.shape, C.shape), D_local(D.shape, D.shape);
    copy_intersection(A, A_local);
    copy_intersection(B, B_local);
    copy_intersection(C, C_local);
    copy_intersection(D, D_local);
    auto A_tile = A_local.get_tile(0), B_tile = B_local.get_tile(0),
         C_tile = C_local.get_tile(0), D_tile = D_local.get_tile(0);
    A_tile.acquire(STARPU_R);
    B_tile.acquire(STARPU_R);
    C_tile.acquire(STARPU_R);
    D_tile.acquire(STARPU_R);
    auto A_tile_ptr = A_tile.get_local_ptr(),
         B_tile_ptr = B_tile.get_local_ptr(),
         C_tile_ptr = C_tile.get_local_ptr(),
         D_tile_ptr = D_tile.get_local_ptr();
    T *tmp_ptr = new T[D_tile.nelems];
    // D = alpha*op(A)*op(B) + beta*C
    // tmp = alpha*op(A)*op(B) + beta*C
    for(Index i = 0; i < D_tile.nelems; ++i)
    {
        tmp_ptr[i] = D_tile_ptr[i];
    }
    D_tile.release();
    T full = norm(M, N, tmp_ptr, ldD);
    T one = 1;
    gemm_naive(transA, transB, M, N, K, -alpha, A_tile_ptr, ldA, B_tile_ptr,
            ldB, one, tmp_ptr, ldD);
    A_tile.release();
    B_tile.release();
    // tmp = beta*C
    if(beta != 0)
    {
        for(Index m = 0; m < M; ++m)
        {
            for(Index n = 0; n < N; ++n)
            {
                Index C_offset = n*ldC + m;
                Index D_offset = n*ldD + m;
                tmp_ptr[D_offset] -= beta * C_tile_ptr[C_offset];
            }
        }
    }
    C_tile.release();
    // D = 0
    T diff = norm(M, N, tmp_ptr, ldD);
    delete[] tmp_ptr;
    // 3 is a magic constant to supress growing rounding errors
    T threshold = 3 * full * std::numeric_limits<T>::epsilon();
    if(diff > threshold)
    {
        std::cout << "diff/threshold=" << diff/threshold << "\n";
        throw std::runtime_error("GEMM is incorrect");
    }
}

template<typename T>
void validate_gemm()
{
    // Traits for different tensors to check operations
    TensorTraits A1_traits({5, 6, 7, 8}, {2, 3, 4, 5}),
                 A1T_traits({8, 5, 6, 7}, {5, 2, 3, 4}),
                 B1_traits({8, 7, 6}, {5, 3, 2}),
                 B1T_traits({7, 6, 8}, {3, 2, 5}),
                 C1_traits({5, 6, 7, 7, 6}, {2, 3, 4, 3, 2}),
                 C1T_traits({7, 6, 5, 6, 7}, {3, 2, 2, 3, 4}),
                 A2_traits({5, 6, 7}, {2, 3, 4}),
                 A2T_traits({6, 7, 5}, {3, 4, 2}),
                 B2_traits({6, 7, 6, 5}, {3, 4, 3, 2}),
                 B2T_traits({6, 5, 6, 7}, {3, 2, 3, 4}),
                 C2_traits({5, 6, 5}, {2, 3, 2}),
                 C2T_traits({6, 5, 5}, {3, 2, 2});
    // Sizes of corresponding matrices
    Index C1M = 5*6*7, C1N = 7*6, C1K = 8, C2M = 5, C2N = 6*5, C2K = 6*7;
    // Construct tensors
    Tensor<T> A1(A1_traits), A1T(A1T_traits),
        B1(B1_traits), B1T(B1T_traits),
        C1(C1_traits), C1T(C1T_traits),
        C1_copy(C1_traits), C1T_copy(C1T_traits),
        A2(A2_traits), A2T(A2T_traits),
        B2(B2_traits), B2T(B2T_traits),
        C2(C2_traits), C2T(C2T_traits),
        C2_copy(C2_traits), C2T_copy(C2T_traits);
    // Randomly init
    unsigned long long A1_seed = 100, A1T_seed = 101,
                  B1_seed = 102, B1T_seed = 103,
                  C1_seed = 104, C1T_seed = 105,
                  A2_seed = 106, A2T_seed = 107,
                  B2_seed = 108, B2T_seed = 109,
                  C2_seed = 110, C2T_seed = 111;
    randn(A1, A1_seed);
    randn(A1T, A1T_seed);
    randn(B1, B1_seed);
    randn(B1T, B1T_seed);
    randn(C1, C1_seed);
    randn(C1T, C1T_seed);
    randn(C1_copy, C1_seed);
    randn(C1T_copy, C1T_seed);
    randn(A2, A2_seed);
    randn(A2T, A2T_seed);
    randn(B2, B2_seed);
    randn(B2T, B2T_seed);
    randn(C2, C2_seed);
    randn(C2T, C2T_seed);
    randn(C2_copy, C2_seed);
    randn(C2T_copy, C2T_seed);
    // Scalar values
    T alpha[3] = {0.0, 1.0, 2.0}, beta[3] = {1.0, 0.0, -2.0};
    // Check gemm with alpha=one and beta=zero
    for(Index i = 0; i < 3; ++i)
    {
        T a = alpha[i], b = beta[i];
        copy_intersection(C1, C1_copy);
        gemm(a, TransOp::NoTrans, A1, TransOp::NoTrans, B1, b, C1, 1);
        test_gemm(TransOp::NoTrans, TransOp::NoTrans, C1M, C1N, C1K, a,
                A1, C1M, B1, C1K, b, C1_copy, C1M, C1, C1M);
        copy_intersection(C1, C1_copy);
        gemm(a, TransOp::NoTrans, A1, TransOp::Trans, B1T, b, C1, 1);
        test_gemm(TransOp::NoTrans, TransOp::Trans, C1M, C1N, C1K, a,
                A1, C1M, B1T, C1N, b, C1_copy, C1M, C1, C1M);
        copy_intersection(C1, C1_copy);
        gemm(a, TransOp::Trans, A1T, TransOp::NoTrans, B1, b, C1, 1);
        test_gemm(TransOp::Trans, TransOp::NoTrans, C1M, C1N, C1K, a,
                A1T, C1K, B1, C1K, b, C1_copy, C1M, C1, C1M);
        copy_intersection(C1, C1_copy);
        gemm(a, TransOp::Trans, A1T, TransOp::Trans, B1T, b, C1, 1);
        test_gemm(TransOp::Trans, TransOp::Trans, C1M, C1N, C1K, a,
                A1T, C1K, B1T, C1N, b, C1_copy, C1M, C1, C1M);
        copy_intersection(C1T, C1T_copy);
        gemm(a, TransOp::NoTrans, B1T, TransOp::NoTrans, A1T, b, C1T, 1);
        test_gemm(TransOp::NoTrans, TransOp::NoTrans, C1N, C1M, C1K, a,
                B1T, C1N, A1T, C1K, b, C1T_copy, C1N, C1T, C1N);
        copy_intersection(C1T, C1T_copy);
        gemm(a, TransOp::NoTrans, B1T, TransOp::Trans, A1, b, C1T, 1);
        test_gemm(TransOp::NoTrans, TransOp::Trans, C1N, C1M, C1K, a,
                B1T, C1N, A1, C1M, b, C1T_copy, C1N, C1T, C1N);
        copy_intersection(C1T, C1T_copy);
        gemm(a, TransOp::Trans, B1, TransOp::NoTrans, A1T, b, C1T, 1);
        test_gemm(TransOp::Trans, TransOp::NoTrans, C1N, C1M, C1K, a,
                B1, C1K, A1T, C1K, b, C1T_copy, C1N, C1T, C1N);
        copy_intersection(C1T, C1T_copy);
        gemm(a, TransOp::Trans, B1, TransOp::Trans, A1, b, C1T, 1);
        test_gemm(TransOp::Trans, TransOp::Trans, C1N, C1M, C1K, a,
                B1, C1K, A1, C1M, b, C1T_copy, C1N, C1T, C1N);
        copy_intersection(C2, C2_copy);
        gemm(a, TransOp::NoTrans, A2, TransOp::NoTrans, B2, b, C2, 2);
        test_gemm(TransOp::NoTrans, TransOp::NoTrans, C2M, C2N, C2K, a,
                A2, C2M, B2, C2K, b, C2_copy, C2M, C2, C2M);
        copy_intersection(C2, C2_copy);
        gemm(a, TransOp::NoTrans, A2, TransOp::Trans, B2T, b, C2, 2);
        test_gemm(TransOp::NoTrans, TransOp::Trans, C2M, C2N, C2K, a,
                A2, C2M, B2T, C2N, b, C2_copy, C2M, C2, C2M);
        copy_intersection(C2, C2_copy);
        gemm(a, TransOp::Trans, A2T, TransOp::NoTrans, B2, b, C2, 2);
        test_gemm(TransOp::Trans, TransOp::NoTrans, C2M, C2N, C2K, a,
                A2T, C2K, B2, C2K, b, C2_copy, C2M, C2, C2M);
        copy_intersection(C2, C2_copy);
        gemm(a, TransOp::Trans, A2T, TransOp::Trans, B2T, b, C2, 2);
        test_gemm(TransOp::Trans, TransOp::Trans, C2M, C2N, C2K, a,
                A2T, C2K, B2T, C2N, b, C2_copy, C2M, C2, C2M);
        copy_intersection(C2T, C2T_copy);
        gemm(a, TransOp::NoTrans, B2T, TransOp::NoTrans, A2T, b, C2T, 2);
        test_gemm(TransOp::NoTrans, TransOp::NoTrans, C2N, C2M, C2K, a,
                B2T, C2N, A2T, C2K, b, C2T_copy, C2N, C2T, C2N);
        copy_intersection(C2T, C2T_copy);
        gemm(a, TransOp::NoTrans, B2T, TransOp::Trans, A2, b, C2T, 2);
        test_gemm(TransOp::NoTrans, TransOp::Trans, C2N, C2M, C2K, a,
                B2T, C2N, A2, C2M, b, C2T_copy, C2N, C2T, C2N);
        copy_intersection(C2T, C2T_copy);
        gemm(a, TransOp::Trans, B2, TransOp::NoTrans, A2T, b, C2T, 2);
        test_gemm(TransOp::Trans, TransOp::NoTrans, C2N, C2M, C2K, a,
                B2, C2K, A2T, C2K, b, C2T_copy, C2N, C2T, C2N);
        copy_intersection(C2T, C2T_copy);
        gemm(a, TransOp::Trans, B2, TransOp::Trans, A2, b, C2T, 2);
        test_gemm(TransOp::Trans, TransOp::Trans, C2N, C2M, C2K, a,
                B2, C2K, A2, C2M, b, C2T_copy, C2N, C2T, C2N);
    }
    // Negative tests
    auto fail_trans_val = static_cast<TransOp::Value>(-1);
    auto fail_trans = *reinterpret_cast<TransOp *>(&fail_trans_val);
    T one = 1;
    TESTN(gemm(one, fail_trans, A1, TransOp::NoTrans, B1, one, C1, 1));
    TESTN(gemm(one, fail_trans, A1, TransOp::Trans, B1T, one, C1, 1));
    TESTN(gemm(one, TransOp::NoTrans, A1, fail_trans, B1, one, C1, 1));
    TESTN(gemm(one, TransOp::Trans, A1T, fail_trans, B1, one, C1, 1));
    TESTN(gemm(one, TransOp::Trans, B2, TransOp::Trans, A2, one, C2T, -1));
    TESTN(gemm(one, TransOp::Trans, B1, TransOp::Trans, A2, one, C2T, 2));
    TESTN(gemm(one, TransOp::Trans, B2, TransOp::Trans, A2, one, C2T, 4));
    TESTN(gemm(one, TransOp::Trans, B1, TransOp::Trans, A1, one, C1T, 4));
    TESTN(gemm(one, TransOp::NoTrans, A1T, TransOp::NoTrans, B1, one, C1, 1));
    TESTN(gemm(one, TransOp::Trans, A1, TransOp::NoTrans, B1, one, C1, 1));
    TESTN(gemm(one, TransOp::NoTrans, A1T, TransOp::Trans, B1T, one, C1, 1));
    TESTN(gemm(one, TransOp::Trans, A1, TransOp::Trans, B1T, one, C1, 1));
    TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, B1, one, C1T, 1));
    TESTN(gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, B1, one, C1T, 1));
    TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::Trans, B1T, one, C1T, 1));
    TESTN(gemm(one, TransOp::Trans, A1T, TransOp::Trans, B1T, one, C1T, 1));
    for(Index i = 0; i < C1.ndim; ++i)
    {
        auto shape = C1.shape, basetile_shape = C1.basetile_shape;
        ++shape[i];
        Tensor<T> tmp(shape, basetile_shape);
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, B1, one, tmp,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, B1, one, tmp,
                    1));
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::Trans, B1T, one, tmp,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::Trans, B1T, one, tmp,
                    1));
        --shape[i];
        ++basetile_shape[i];
        Tensor<T> tmp2(shape, basetile_shape);
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, B1, one, tmp2,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, B1, one, tmp2,
                    1));
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::Trans, B1T, one, tmp2,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::Trans, B1T, one, tmp2,
                    1));
    }
    for(Index i = 0; i < B1.ndim; ++i)
    {
        auto shape = B1.shape, basetile_shape = B1.basetile_shape;
        ++shape[i];
        Tensor<T> tmp(shape, basetile_shape);
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, tmp, one, C1,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, tmp, one, C1,
                    1));
        auto shapeT = B1T.shape, basetile_shapeT = B1T.basetile_shape;
        ++shapeT[i];
        Tensor<T> tmpT(shapeT, basetile_shapeT);
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::Trans, tmpT, one, C1,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::Trans, tmpT, one, C1,
                    1));
        --shape[i];
        ++basetile_shape[i];
        Tensor<T> tmp2(shape, basetile_shape);
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, tmp2, one, C1,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, tmp2, one, C1,
                    1));
        --shapeT[i];
        ++basetile_shapeT[i];
        Tensor<T> tmp2T(shapeT, basetile_shapeT);
        TESTN(gemm(one, TransOp::NoTrans, A1, TransOp::Trans, tmp2T, one, C1,
                    1));
        TESTN(gemm(one, TransOp::Trans, A1T, TransOp::Trans, tmp2T, one, C1,
                    1));
    }
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_gemm<float>();
    validate_gemm<double>();
    return 0;
}

