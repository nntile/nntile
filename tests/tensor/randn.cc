#include "nntile/tensor/randn.hh"
#include "check_tensors_intersection.hh"

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tensor<T> A({4, 5, 6, 7}, {1, 2, 3, 4}), B(A.shape, A.shape);
    unsigned long long seed = 100;
    randn(A, seed);
    randn(A, B, {1, 1, 1, 1}, seed);
    check_tensors_intersection(A, {0, 0, 0, 0}, B, {1, 1, 1, 1});
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

