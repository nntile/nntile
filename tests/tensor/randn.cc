#include "nntile/tensor/randn.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tensor<T> A({4, 5, 6, 7}, {1, 2, 3, 4}), B(A.shape, A.shape);
    unsigned long long seed = 100;
    randn(A, seed);
    TESTN(randn(A, B, {0}, seed));
    TESTN(randn(A, B, {0, 0, 0, 0, 0}, seed));
    randn(A, B, {1, 1, 1, 1}, seed);
    check_tensors_intersection(A, {0, 0, 0, 0}, B, {1, 1, 1, 1});
    randn(A, B, {14, 0, 0, 0}, seed);
    check_tensors_intersection(A, {0, 0, 0, 0}, B, {14, 0, 0, 0});
    Tensor<T> C({1, 2, 3}, {1, 2, 3});
    TESTN(randn(A, C, {0, 0, 0, 0}, seed));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

