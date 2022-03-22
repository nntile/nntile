#include <nntile.hh>

using namespace nntile;

int main(int argc, char **argv)
{
    using T = float;
    (void *)starpu_init(nullptr);
    std::array<int, 4> shape({3, 2, 1, 10});
    std::vector<int> vshape({3, 2, 1, 10});
    ContiguousTileTraits traits(shape);
    ContiguousTileTraits traitsA({3, 2, 1, 10});
    ContiguousTileTraits traitsAT({10, 1, 2, 3});
    ContiguousTileTraits traitsB({10, 5, 6});
    ContiguousTileTraits traitsBT({6, 5, 10});
    ContiguousTileTraits traitsB2({1, 10, 5, 6});
    ContiguousTileTraits traitsB2T({6, 5, 10, 1});
    ContiguousTileTraits traitsC({3, 2, 1, 5, 6});
    ContiguousTileTraits traitsCT({6, 5, 1, 2, 3});
    ContiguousTileTraits traitsC2({3, 2, 5, 6});
    ContiguousTileTraits traitsC2T({6, 5, 2, 3});
    auto *ptrA = new T[traitsA.nelems];
    auto *ptrB = new T[traitsB.nelems];
    auto *ptrC = new T[traitsC.nelems];
    std::cout << traitsA.matrix_shape.size() << " ";
    std::cout << traitsA.matrix_shape[0][0] << " ";
    std::cout << traitsA.matrix_shape[1][0] << " ";
    std::cout << traitsA.matrix_shape[2][0] << " ";
    std::cout << traitsA.matrix_shape[0][1] << " ";
    std::cout << traitsA.matrix_shape[1][1] << " ";
    std::cout << traitsA.matrix_shape[2][1] << " ";
    std::cout << "\n";
    std::cout << traitsB.matrix_shape.size() << " ";
    std::cout << traitsB.matrix_shape[0][0] << " ";
    std::cout << traitsB.matrix_shape[1][0] << " ";
    std::cout << traitsB.matrix_shape[0][1] << " ";
    std::cout << traitsB.matrix_shape[1][1] << " ";
    std::cout << "\n";
    std::cout << traitsC.matrix_shape.size() << " ";
    std::cout << traitsC.matrix_shape[0][0] << " ";
    std::cout << traitsC.matrix_shape[1][0] << " ";
    std::cout << traitsC.matrix_shape[2][0] << " ";
    std::cout << traitsC.matrix_shape[3][0] << " ";
    std::cout << traitsC.matrix_shape[0][1] << " ";
    std::cout << traitsC.matrix_shape[1][1] << " ";
    std::cout << traitsC.matrix_shape[2][1] << " ";
    std::cout << traitsC.matrix_shape[3][1] << " ";
    std::cout << "\n";
    ContiguousTile<T> tilenull(traitsA, nullptr);
    ContiguousTile<T> tileA(traitsA, ptrA);
    ContiguousTile<T> tileAT(traitsAT, ptrA);
    ContiguousTile<T> tileB(traitsB, ptrB);
    ContiguousTile<T> tileBT(traitsBT, ptrB);
    ContiguousTile<T> tileB2(traitsB2, ptrB);
    ContiguousTile<T> tileB2T(traitsB2T, ptrB);
    ContiguousTile<T> tileC(traitsC, ptrC);
    ContiguousTile<T> tileCT(traitsCT, ptrC);
    ContiguousTile<T> tileC2(traitsC2, ptrC);
    ContiguousTile<T> tileC2T(traitsC2T, ptrC);
    tileA.data_register();
    tileAT.data_register();
    tileB.data_register();
    tileBT.data_register();
    tileC.data_register();
    tileCT.data_register();
    tileB2.data_register();
    tileB2T.data_register();
    tileC2.data_register();
    tileC2T.data_register();
    T alpha = 1.0, beta = 0.0;
    gemm_check_ndim(tileA, tileB, tileC);
    gemm_check(TransOp::NoTrans, tileA, TransOp::NoTrans, tileB, tileC);
    gemm_check(TransOp::NoTrans, tileA, TransOp::Trans, tileBT, tileC);
    gemm_check(TransOp::Trans, tileAT, TransOp::NoTrans, tileB, tileC);
    gemm_check(TransOp::Trans, tileAT, TransOp::Trans, tileBT, tileC);
    gemm_check(TransOp::NoTrans, tileBT, TransOp::NoTrans, tileAT, tileCT);
    gemm_check(TransOp::NoTrans, tileBT, TransOp::Trans, tileA, tileCT);
    gemm_check(TransOp::Trans, tileB, TransOp::NoTrans, tileAT, tileCT);
    gemm_check(TransOp::Trans, tileB, TransOp::Trans, tileA, tileCT);
    gemm_check(TransOp::NoTrans, tileA, TransOp::NoTrans, tileB2, tileC2, 2);
    gemm_check(TransOp::NoTrans, tileA, TransOp::Trans, tileB2T, tileC2, 2);
    gemm_check(TransOp::Trans, tileAT, TransOp::NoTrans, tileB2, tileC2, 2);
    gemm_check(TransOp::Trans, tileAT, TransOp::Trans, tileB2T, tileC2, 2);
    gemm_check(TransOp::NoTrans, tileB2T, TransOp::NoTrans, tileAT, tileC2T, 2);
    gemm_check(TransOp::NoTrans, tileB2T, TransOp::Trans, tileA, tileC2T, 2);
    gemm_check(TransOp::Trans, tileB2, TransOp::NoTrans, tileAT, tileC2T, 2);
    gemm_check(TransOp::Trans, tileB2, TransOp::Trans, tileA, tileC2T, 2);
    gemm(alpha, TransOp::NoTrans, tileA, TransOp::NoTrans, tileB, beta,
            tileC, 1, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileA, TransOp::Trans, tileBT, beta,
            tileC, 1, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileAT, TransOp::NoTrans, tileB, beta,
            tileC, 1, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileAT, TransOp::Trans, tileBT, beta,
            tileC, 1, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileBT, TransOp::NoTrans, tileAT, beta,
            tileCT, 1, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileBT, TransOp::Trans, tileA, beta,
            tileCT, 1, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileB, TransOp::NoTrans, tileAT, beta,
            tileCT, 1, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileB, TransOp::Trans, tileA, beta,
            tileCT, 1, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileA, TransOp::NoTrans, tileB2, beta,
            tileC2, 2, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileA, TransOp::Trans, tileB2T, beta,
            tileC2, 2, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileAT, TransOp::NoTrans, tileB2, beta,
            tileC2, 2, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileAT, TransOp::Trans, tileB2T, beta,
            tileC2, 2, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileB2T, TransOp::NoTrans, tileAT, beta,
            tileC2T, 2, Debug::Debug);
    gemm(alpha, TransOp::NoTrans, tileB2T, TransOp::Trans, tileA, beta,
            tileC2T, 2, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileB2, TransOp::NoTrans, tileAT, beta,
            tileC2T, 2, Debug::Debug);
    gemm(alpha, TransOp::Trans, tileB2, TransOp::Trans, tileA, beta,
            tileC2T, 2, Debug::Debug);
    gelu(tileC);
    gelu(tileC2);
    ContiguousTileTraits traits_biasA0({2, 1, 10});
    ContiguousTileTraits traits_biasA1({3, 1, 10});
    ContiguousTileTraits traits_biasA2({3, 2, 10});
    ContiguousTileTraits traits_biasA3({3, 2, 1});
    T *ptr_biasA0 = new T[traits_biasA0.nelems];
    T *ptr_biasA1 = new T[traits_biasA1.nelems];
    T *ptr_biasA2 = new T[traits_biasA2.nelems];
    T *ptr_biasA3 = new T[traits_biasA3.nelems];
    ContiguousTile<T> biasA0(traits_biasA0, ptr_biasA0);
    ContiguousTile<T> biasA1(traits_biasA1, ptr_biasA1);
    ContiguousTile<T> biasA2(traits_biasA2, ptr_biasA2);
    ContiguousTile<T> biasA3(traits_biasA3, ptr_biasA3);
    biasA0.data_register();
    biasA1.data_register();
    biasA2.data_register();
    biasA3.data_register();
    bias(tileA, biasA0, 0);
    bias(tileA, biasA1, 1);
    bias(tileA, biasA2, 2);
    bias(tileA, biasA3, 3);
    tileA.data_unregister();
    tileAT.data_unregister();
    tileB.data_unregister();
    tileBT.data_unregister();
    tileC.data_unregister();
    tileCT.data_unregister();
    tileB2.data_unregister();
    tileB2T.data_unregister();
    tileC2.data_unregister();
    tileC2T.data_unregister();
    biasA0.data_unregister();
    biasA1.data_unregister();
    biasA2.data_unregister();
    biasA3.data_unregister();
    starpu_shutdown();
    delete[] ptrA;
    delete[] ptrB;
    delete[] ptrC;
    delete[] ptr_biasA0;
    delete[] ptr_biasA1;
    delete[] ptr_biasA2;
    delete[] ptr_biasA3;
}

