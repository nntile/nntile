#include "nntile/tensor/copy.hh"
#include "nntile/tensor/randn.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_copy()
{
    Tensor<T> scalar({}, {}), scalar2({}, {});
    Tensor<T> A({4, 5, 6}, {1, 2, 3}), B(A.shape, A.shape),
        C({5, 6, 7}, {3, 4, 5});
    unsigned long long seed = 100000000000UL;
    randn(scalar, seed);
    randn(scalar2, seed*seed);
    copy_async(scalar, scalar2);
    starpu_task_wait_for_all();
    TESTA(check_tensors_intersection(scalar, scalar2));
    TESTA(check_tensors_intersection(scalar2, scalar));
    TESTN(copy(scalar, {0}, scalar2, {}));
    TESTN(copy(scalar, {}, scalar2, {0}));
    TESTN(copy(scalar, {}, Tensor<T>({1}, {1}), {0}));
    randn(A, seed);
    randn(B, seed*seed+1);
    randn(C, (seed*seed+1)*seed+1);
    TESTA(!check_tensors_intersection(A, {0, 0, 0}, B, {0, 0, 0}));
    TESTN(copy(A, {0, 0, 0, 0}, B, {0}));
    TESTN(copy(A, {0, 0, 0, 0}, B, {0, 0, 0, 0, 0}));
    TESTN(copy(A, {0, 0, 0, 0, 0}, B, {0, 0, 0, 0}));
    copy(A, {1, 1, 1}, B, {0, 0, 0});
    TESTA(check_tensors_intersection(A, {1, 1, 1}, B, {0, 0, 0}));
    TESTA(check_tensors_intersection(B, {0, 0, 0}, A, {1, 1, 1}));
    copy(A, {0, 0, 0}, C, {0, 0, 0});
    TESTA(check_tensors_intersection(A, {0, 0, 0}, C, {0, 0, 0}));
    TESTA(check_tensors_intersection(C, {0, 0, 0}, A, {0, 0, 0}));
    copy(A, {0, 0, 0}, C, {1, 2, 3});
    TESTA(check_tensors_intersection(A, {0, 0, 0}, C, {1, 2, 3}));
    TESTA(check_tensors_intersection(C, {1, 2, 3}, A, {0, 0, 0}));
    copy(A, {2, 1, 3}, C, {1, 2, 3});
    TESTA(check_tensors_intersection(A, {2, 1, 3}, C, {1, 2, 3}));
    TESTA(check_tensors_intersection(C, {1, 2, 3}, A, {2, 1, 3}));
    copy(B, {1, 1, 1}, A, {0, 0, 0});
    TESTA(check_tensors_intersection(A, {0, 0, 0}, B, {1, 1, 1}));
    TESTA(check_tensors_intersection(B, {1, 1, 1}, A, {0, 0, 0}));
    copy(A, {0, 0, 0}, B, {1, 1, 1});
    TESTA(check_tensors_intersection(A, {0, 0, 0}, B, {1, 1, 1}));
    TESTA(check_tensors_intersection(B, {1, 1, 1}, A, {0, 0, 0}));
    copy(A, {0, 0, 0}, B, {3, 4, 5});
    TESTA(!check_tensors_intersection(A, {0, 0, 0}, B, {0, 0, 0}));
    TESTA(!check_tensors_intersection(B, {0, 0, 0}, A, {0, 0, 0}));
    Tensor<T> D({1, 2, 3, 4}, {1, 2, 3, 4});
    TESTN(copy(A, D));
    TESTN(copy(A, {0, 0, 0}, D, {0, 0, 0, 0}));
    Tensor<T> AA(A.shape, A.basetile_shape);
    copy(A, AA);
    TESTA(check_tensors_intersection(A, {0, 0, 0}, AA, {0, 0, 0}));
    TESTA(check_tensors_intersection(AA, {0, 0, 0}, A, {0, 0, 0}));
    copy(A, {0, 0, 0}, AA, A.basetile_shape);
    TESTA(!check_tensors_intersection(A, {0, 0, 0}, AA, {0, 0, 0}));
    TESTA(!check_tensors_intersection(AA, {0, 0, 0}, A, {0, 0, 0}));
    TESTA(check_tensors_intersection(A, {0, 0, 0}, AA, A.basetile_shape));
    TESTA(check_tensors_intersection(AA, A.basetile_shape, A, {0, 0, 0}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_copy<float>();
    validate_copy<double>();
    return 0;
}

