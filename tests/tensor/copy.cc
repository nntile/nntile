#include "nntile/tensor/copy.hh"
#include "nntile/tensor/randn.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_copy()
{
    Tensor<T> scalar({}, {}), scalar2({}, {});
    Tensor<T> A({4, 5, 6}, {1, 2, 3}), B(A.shape, A.shape);
    unsigned long long seed = 100000000000UL;
    randn(scalar, seed);
    randn(scalar2, seed*seed);
    copy_intersection_async(scalar, scalar2);
    starpu_task_wait_for_all();
    TESTA(check_tensors_intersection(scalar, scalar2));
    TESTA(check_tensors_intersection(scalar2, scalar));
    TESTN(copy_intersection(scalar, {0}, scalar2, {}));
    TESTN(copy_intersection(scalar, {}, scalar2, {0}));
    TESTN(copy_intersection(scalar, {}, Tensor<T>({1}, {1}), {0}));
    randn(A, seed);
    randn(B, seed*seed+1);
    TESTA(!check_tensors_intersection(A, {0, 0, 0}, B, {0, 0, 0}));
    TESTN(copy_intersection(A, {0, 0, 0, 0}, B, {0}));
    TESTN(copy_intersection(A, {0, 0, 0, 0}, B, {0, 0, 0, 0, 0}));
    TESTN(copy_intersection(A, {0, 0, 0, 0, 0}, B, {0, 0, 0, 0}));
    copy_intersection(A, {0, 0, 0}, B, {1, 1, 1});
    TESTA(check_tensors_intersection(A, {0, 0, 0}, B, {1, 1, 1}));
    TESTA(check_tensors_intersection(B, {1, 1, 1}, A, {0, 0, 0}));
    copy_intersection(A, {0, 0, 0}, B, {3, 4, 5});
    TESTA(!check_tensors_intersection(A, {0, 0, 0}, B, {0, 0, 0}));
    TESTA(!check_tensors_intersection(B, {0, 0, 0}, A, {0, 0, 0}));
    Tensor<T> C({1, 2, 3, 4}, {1, 2, 3, 4});
    TESTN(copy_intersection(A, C));
    TESTN(copy_intersection(A, {0, 0, 0}, C, {0, 0, 0, 0}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_copy<float>();
    validate_copy<double>();
    return 0;
}

