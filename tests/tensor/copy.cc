#include "nntile/tensor/copy.hh"
#include "nntile/tensor/randn.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_copy()
{
//    Tensor<T> A({2, 3}, {1, 2}), B(A.shape, A.shape);
//    unsigned long long seed = 100;
//    randn(A, seed);
//    randn(B, 2*seed);
//    auto B_tile = B.get_tile(0);
//    B_tile.acquire(STARPU_R);
//    auto B_ptr = B_tile.get_local_ptr();
//    for(size_t i = 0; i < B_tile.nelems; ++i)
//    {
//        std::cout << B_ptr[i] << " ";
//    }
//    std::cout << "\n";
//    B_tile.release();
//    copy(A, {0, 0}, B, {1, 1});
//    for(size_t it = 0; it < A.grid.nelems; ++it)
//    {
//        auto A_tile = A.get_tile(it);
//        A_tile.acquire(STARPU_R);
//        auto A_ptr = A_tile.get_local_ptr();
//        for(size_t i = 0; i < A_tile.nelems; ++i)
//        {
//            std::cout << A_ptr[i] << " ";
//        }
//        std::cout << "\n";
//        A_tile.release();
//    }
//    B_tile.acquire(STARPU_R);
//    {
//        auto B_tile = B.get_tile(0);
//        auto B_ptr = B_tile.get_local_ptr();
//        for(size_t i = 0; i < B_tile.nelems; ++i)
//        {
//            std::cout << B_ptr[i] << " ";
//        }
//        std::cout << "\n";
//    }
//    B_tile.release();
//    check_tensors_intersection(A, {0, 0}, B, {1, 1});
    {
    Tensor<T> A({4, 5, 6}, {1, 2, 3}), B(A.shape, A.shape);
    unsigned long long seed = 100;
    randn(A, seed);
    randn(B, seed+1);
//    TESTN(copy(A, {0, 0, 0, 0}, B, {0}));
//    TESTN(copy(A, {0, 0, 0, 0}, B, {0, 0, 0, 0, 0}));
//    TESTN(copy(A, {0, 0, 0, 0, 0}, B, {0, 0, 0, 0}));
    copy(A, {0, 0, 0}, B, {1, 1, 1});
    check_tensors_intersection(A, {0, 0, 0}, B, {1, 1, 1});
//    return;
//    copy(A, {0, 0, 0, 0}, B, {14, 0, 0, 0});
//    check_tensors_intersection(A, {0, 0, 0, 0}, B, {14, 0, 0, 0});
//    Tensor<T> C({1, 2, 3}, {1, 2, 3});
//    TESTN(copy(A, C));
//    TESTN(copy(A, {0, 0, 0, 0}, C, {0, 0, 0, 0}));
    }
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_copy<float>();
    validate_copy<double>();
    return 0;
}

