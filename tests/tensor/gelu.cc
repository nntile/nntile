#include "nntile/tensor/gelu.hh"
#include "nntile/tile/gelu.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_gelu(const Tensor<T> &A)
{
    Tensor<T> A_local(A.shape, A.shape);
    copy(A, A_local);
    gelu(A);
    TESTA(!check_tensors_intersection(A, A_local));
    TESTA(!check_tensors_intersection(A_local, A));
    gelu(A_local.get_tile(0));
    TESTA(check_tensors_intersection(A, A_local));
    TESTA(check_tensors_intersection(A_local, A));
}

template<typename T>
void validate_gelu()
{
    Tensor<T> scalar({}, {}), A({4, 5, 6, 3}, {2, 3, 3, 2});
    unsigned long long seed = 100000000000000001ULL;
    randn(scalar, seed);
    check_gelu(scalar);
    randn(A, seed);
    check_gelu(A);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_gelu<float>();
    validate_gelu<double>();
    return 0;
}

