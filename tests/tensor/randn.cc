#include "nntile/tensor/randn.hh"
#include "check_tensors_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tensor<T> scalar({}, {}), scalar2({}, {});
    T one = 1, zero = 0;
    constexpr unsigned long long seed = 100000000000001ULL;
    randn(scalar, {}, {}, {}, seed);
    randn(scalar2, {}, {}, {}, seed);
    TESTA(check_tensors_intersection(scalar, scalar2));
    randn(scalar2, seed*seed);
    TESTA(!check_tensors_intersection(scalar, scalar2));
    Tensor<T> big({5, 6, 7, 8}, {2, 3, 4, 5}),
        small({2, 2, 2, 2}, {1, 2, 2, 1});
    randn_async(big, seed);
    starpu_task_wait_for_all();
    randn(small, {1, 2, 3, 2}, big.shape, big.stride, seed);
    TESTA(check_tensors_intersection(big, {0, 0, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(check_tensors_intersection(small, {1, 2, 3, 2}, big, {0, 0, 0, 0}));
    TESTA(!check_tensors_intersection(big, {1, 0, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tensors_intersection(big, {0, 1, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tensors_intersection(big, {0, 0, 1, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tensors_intersection(big, {0, 0, 0, 1}, small, {1, 2, 3, 2}));
    TESTA(!check_tensors_intersection(small, {1, 2, 3, 2}, big, {1, 0, 0, 0}));
    TESTA(!check_tensors_intersection(small, {1, 2, 3, 2}, big, {0, 1, 0, 0}));
    TESTA(!check_tensors_intersection(small, {1, 2, 3, 2}, big, {0, 0, 1, 0}));
    TESTA(!check_tensors_intersection(small, {1, 2, 3, 2}, big, {0, 0, 0, 1}));
    TESTN(randn(small, {4, 0, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 5, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 6, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 0, 7}, big.shape, big.stride, seed));
    TESTN(randn(small, {-1, 0, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, -1, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, -1, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 0, -1}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 0, 0}, {5, 6, 7}, big.stride, seed));
    TESTN(randn(small, {0, 0, 0, -1}, big.shape, {5, 30, 210}, seed));
    std::vector<Index> stride(big.stride);
    ++stride[0];
    TESTN(randn(small, {0, 0, 0, 0}, big.shape, stride, seed));
    for(int i = 1; i < stride.size(); ++i)
    {
        --stride[i-1];
        ++stride[i];
        TESTN(randn(small, {0, 0, 0, 0}, big.shape, stride, seed));
    }
    TESTA(stride != big.stride);
    Tensor<T> small2({3, 3, 3, 3}, {2, 3, 2, 3});
    randn(small2, {1, 1, 1, 1}, big.shape, big.stride, seed);
    TESTA(check_tensors_intersection(small2, {1, 1, 1, 1}, big,
                {0, 0, 0, 0}));
    TESTA(check_tensors_intersection(small, {1, 2, 3, 2}, small2,
                {1, 1, 1, 1}));
    TESTA(check_tensors_intersection(small2, {1, 1, 1, 1}, small,
                {1, 2, 3, 2}));
    auto small_local = small.get_tile(0).acquire(STARPU_RW);
    small_local[small.get_tile(0).nelems-1] = 0;
    small_local.release();
    TESTA(!check_tensors_intersection(big, {0, 0, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tensors_intersection(small, {1, 2, 3, 2}, big, {0, 0, 0, 0}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

