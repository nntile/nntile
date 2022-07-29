#include "nntile/tensor/norm.hh"
#include "nntile/tile/norm.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "../testing.hh"
#include "check_tensors_intersection.hh"
#include <cmath>

template<typename T>
void check_avg_dev(const Tensor<T> &sum_ssq, const Tensor<T> &sum_ssq_tile,
        const Tensor<T> &avg_dev, const Tensor<T> &avg_dev_tile,
        Index nelems, T eps)
{
    norm_avg_dev(sum_ssq, avg_dev, nelems, eps);
    norm_avg_dev(sum_ssq_tile, avg_dev_tile, nelems, eps);
    TESTA(check_tensors_intersection(avg_dev, avg_dev_tile));
}

template<typename T>
void validate_avg_dev()
{
    Tensor<T> A({9, 10, 13, 15}, {4, 5, 6, 7}), A_tile(A.shape, A.shape);
    constexpr unsigned long long seed = 100000000000001ULL;
    constexpr T eps0 = 0, eps1 = 0.01, eps2=1e+10;
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    for(Index i = 0; i < A.ndim; ++i)
    {
        std::vector<Index> shape(A.ndim), basetile(A.ndim);
        shape[0] = 3;
        basetile[0] = 3;
        Index k = 0;
        Index nelems = A.shape[i];
        for(Index j = 0; j < i; ++j)
        {
            shape[j+1] = A.shape[j];
            basetile[j+1] = A.basetile_shape[j];
        }
        for(Index j = i+1; j < A.ndim; ++j)
        {
            shape[j] = A.shape[j];
            basetile[j] = A.basetile_shape[j];
        }
        std::vector<Index> shape2(shape), basetile2(basetile);
        shape2[0] = 2;
        basetile2[0] = 2;
        Tensor<T> sum_ssq(shape, basetile), sum_ssq_tile(shape, shape);
        Tensor<T> avg_dev(shape2, basetile2), avg_dev_tile(shape2, shape2);
        norm_sum_ssq(A, sum_ssq, i);
        copy(sum_ssq, sum_ssq_tile);
        check_avg_dev(sum_ssq, sum_ssq_tile, avg_dev, avg_dev_tile, nelems,
                eps0);
        check_avg_dev(sum_ssq, sum_ssq_tile, avg_dev, avg_dev_tile, nelems,
                eps1);
        check_avg_dev(sum_ssq, sum_ssq_tile, avg_dev, avg_dev_tile, nelems,
                eps2);
    }
//    TESTN(norm_avg_dev(Tensor<T>({3}), Tensor<T>({2}), -1, T{0}));
//    TESTN(norm_avg_dev(Tensor<T>({3}), Tensor<T>({2}), 0, T{0}));
//    TESTN(norm_avg_dev(Tensor<T>({3}), Tensor<T>({2}), 1, T{-1}));
//    TESTN(norm_avg_dev(Tensor<T>({2}), Tensor<T>({2}), 1, T{1}));
//    TESTN(norm_avg_dev(Tensor<T>({3}), Tensor<T>({3}), 1, T{1}));
//    TESTN(norm_avg_dev(Tensor<T>({3}), Tensor<T>({2, 3}), 1, T{1}));
//    TESTN(norm_avg_dev(Tensor<T>({}), Tensor<T>({}), 1, T{1}));
//    TESTN(norm_avg_dev(Tensor<T>({3, 4}), Tensor<T>({2, 3}), 1, T{1}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_sum_ssq<fp32_t>();
    validate_sum_ssq<fp64_t>();
    validate_avg_dev<fp32_t>();
    validate_avg_dev<fp64_t>();
    return 0;
}

