#include <nntile.hh>

using namespace nntile;

int main(int argc, char **argv)
{
    using T = float;
    StarPU starpu;
    {
    std::vector<int> shape({1000, 11, 12, 1300}),
        tile_shape({512, 4, 4, 512});
    TensorTraits A_traits(shape, tile_shape);
    std::cout << A_traits;
    // Check if destructor does not deregister data twice
    {
        Tensor<T> A(A_traits), B(A), C(Tensor<T>{A_traits});
    }
    TensorTraits A1_traits({11, 12, 13, 14}, {5, 4, 6, 8}),
                 A1T_traits({14, 11, 12, 13}, {8, 5, 4, 6}),
                 B1_traits({14, 15, 16}, {8, 6, 4}),
                 B1T_traits({15, 16, 14}, {6, 4, 8}),
                 C1_traits({11, 12, 13, 15, 16}, {5, 4, 6, 6, 4}),
                 C1T_traits({15, 16, 11, 12, 13}, {6, 4, 5, 4, 6}),
                 A2_traits({8, 9, 10}, {2, 3, 4}),
                 A2T_traits({9, 10, 8}, {3, 4, 2}),
                 B2_traits({9, 10, 11, 12}, {3, 4, 5, 6}),
                 B2T_traits({11, 12, 9, 10}, {5, 6, 3, 4}),
                 C2_traits({8, 11, 12}, {2, 5, 6}),
                 C2T_traits({11, 12, 8}, {5, 6, 2});
    // High-level API for tiles
    Tensor<T> A1(A1_traits),
        A1T(A1T_traits),
        B1(B1_traits),
        B1T(B1T_traits),
        C1(C1_traits),
        C1T(C1T_traits),
        A2(A2_traits),
        A2T(A2T_traits),
        B2(B2_traits),
        B2T(B2T_traits),
        C2(C2_traits),
        C2T(C2T_traits);
    std::cout << "B1T\n" << B1T_traits;
    T one = 1.0, zero = 0.0;
    // Check if tensors match gemm operation
    gemm_check(TransOp::NoTrans, A1, TransOp::NoTrans, B1, C1);
    gemm_check(TransOp::NoTrans, A1, TransOp::Trans, B1T, C1);
    gemm_check(TransOp::Trans, A1T, TransOp::NoTrans, B1, C1);
    gemm_check(TransOp::Trans, A1T, TransOp::Trans, B1T, C1);
    gemm_check(TransOp::NoTrans, B1T, TransOp::NoTrans, A1T, C1T);
    gemm_check(TransOp::NoTrans, B1T, TransOp::Trans, A1, C1T);
    gemm_check(TransOp::Trans, B1, TransOp::NoTrans, A1T, C1T);
    gemm_check(TransOp::Trans, B1, TransOp::Trans, A1, C1T);
    gemm_check(TransOp::NoTrans, A2, TransOp::NoTrans, B2, C2, 2);
    gemm_check(TransOp::NoTrans, A2, TransOp::Trans, B2T, C2, 2);
    gemm_check(TransOp::Trans, A2T, TransOp::NoTrans, B2, C2, 2);
    gemm_check(TransOp::Trans, A2T, TransOp::Trans, B2T, C2, 2);
    gemm_check(TransOp::NoTrans, B2T, TransOp::NoTrans, A2T, C2T, 2);
    gemm_check(TransOp::NoTrans, B2T, TransOp::Trans, A2, C2T, 2);
    gemm_check(TransOp::Trans, B2, TransOp::NoTrans, A2T, C2T, 2);
    gemm_check(TransOp::Trans, B2, TransOp::Trans, A2, C2T, 2);
    // Check gemm with alpha=one and beta=zero
    gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, B1, zero, C1, 1);
    gemm(one, TransOp::NoTrans, A1, TransOp::Trans, B1T, zero, C1, 1);
    gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, B1, zero, C1, 1);
    gemm(one, TransOp::Trans, A1T, TransOp::Trans, B1T, zero, C1, 1);
    gemm(one, TransOp::NoTrans, B1T, TransOp::NoTrans, A1T, zero, C1T, 1);
    gemm(one, TransOp::NoTrans, B1T, TransOp::Trans, A1, zero, C1T, 1);
    gemm(one, TransOp::Trans, B1, TransOp::NoTrans, A1T, zero, C1T, 1);
    gemm(one, TransOp::Trans, B1, TransOp::Trans, A1, zero, C1T, 1);
    gemm(one, TransOp::NoTrans, A2, TransOp::NoTrans, B2, zero, C2, 2);
    gemm(one, TransOp::NoTrans, A2, TransOp::Trans, B2T, zero, C2, 2);
    gemm(one, TransOp::Trans, A2T, TransOp::NoTrans, B2, zero, C2, 2);
    gemm(one, TransOp::Trans, A2T, TransOp::Trans, B2T, zero, C2, 2);
    gemm(one, TransOp::NoTrans, B2T, TransOp::NoTrans, A2T, zero, C2T, 2);
    gemm(one, TransOp::NoTrans, B2T, TransOp::Trans, A2, zero, C2T, 2);
    gemm(one, TransOp::Trans, B2, TransOp::NoTrans, A2T, zero, C2T, 2);
    gemm(one, TransOp::Trans, B2, TransOp::Trans, A2, zero, C2T, 2);
    // Check gemm with alpha=one and beta=one
    gemm(one, TransOp::NoTrans, A1, TransOp::NoTrans, B1, one, C1, 1);
    gemm(one, TransOp::NoTrans, A1, TransOp::Trans, B1T, one, C1, 1);
    gemm(one, TransOp::Trans, A1T, TransOp::NoTrans, B1, one, C1, 1);
    gemm(one, TransOp::Trans, A1T, TransOp::Trans, B1T, one, C1, 1);
    gemm(one, TransOp::NoTrans, B1T, TransOp::NoTrans, A1T, one, C1T, 1);
    gemm(one, TransOp::NoTrans, B1T, TransOp::Trans, A1, one, C1T, 1);
    gemm(one, TransOp::Trans, B1, TransOp::NoTrans, A1T, one, C1T, 1);
    gemm(one, TransOp::Trans, B1, TransOp::Trans, A1, one, C1T, 1);
    gemm(one, TransOp::NoTrans, A2, TransOp::NoTrans, B2, one, C2, 2);
    gemm(one, TransOp::NoTrans, A2, TransOp::Trans, B2T, one, C2, 2);
    gemm(one, TransOp::Trans, A2T, TransOp::NoTrans, B2, one, C2, 2);
    gemm(one, TransOp::Trans, A2T, TransOp::Trans, B2T, one, C2, 2);
    gemm(one, TransOp::NoTrans, B2T, TransOp::NoTrans, A2T, one, C2T, 2);
    gemm(one, TransOp::NoTrans, B2T, TransOp::Trans, A2, one, C2T, 2);
    gemm(one, TransOp::Trans, B2, TransOp::NoTrans, A2T, one, C2T, 2);
    gemm(one, TransOp::Trans, B2, TransOp::Trans, A2, one, C2T, 2);
    }
    return 0;
}
