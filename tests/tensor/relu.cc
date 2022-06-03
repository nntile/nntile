#include "nntile/tensor/relu.hh"
#include "nntile/tile/relu.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_relu(const Tensor<T> &A)
{
    Tensor<T> A_local(A.shape, A.shape);
    copy_intersection(A, A_local);
    return;
    relu(A);
    TESTA(!check_tensors_intersection(A, A_local));
    TESTA(!check_tensors_intersection(A_local, A));
    relu(A_local.get_tile(0));
    TESTA(check_tensors_intersection(A, A_local));
    TESTA(check_tensors_intersection(A_local, A));
}

template<typename T>
void validate_relu()
{
    Tensor<T> scalar({}, {}), A({4, 5, 6, 3}, {2, 3, 3, 2});
    unsigned long long seed = std::numeric_limits<unsigned long long>::max();
    randn(scalar, seed);
    check_relu(scalar);
    randn(A, seed);
    check_relu(A);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_relu<float>();
    validate_relu<double>();
    return 0;
}

