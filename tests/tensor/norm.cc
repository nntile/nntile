#include "nntile/tensor/norm.hh"
#include "nntile/tile/norm.hh"
#include "nntile/tensor/randn.hh"
#include "nntile/tensor/copy.hh"
#include "../testing.hh"
#include "check_tensors_intersection.hh"
#include <cmath>

using namespace nntile;

template<typename T>
void check_sum_ssq(const Tensor<T> &src, const Tensor<T> &src_tile,
        const Tensor<T> &sum_ssq, const Tensor<T> &sum_ssq_work,
        const std::vector<Index> &axes)
{
    norm_sum_ssq(src, sum_ssq, sum_ssq_work, axes);
    Tensor<T> sum_ssq_tile(sum_ssq.shape, sum_ssq.shape);
    norm_sum_ssq(src_tile.get_tile(0), sum_ssq_tile.get_tile(0), axes);
    Tensor<T> sum_ssq_tile2(sum_ssq.shape, sum_ssq.shape);
    copy_intersection(sum_ssq, sum_ssq_tile2);
    auto tile = sum_ssq_tile.get_tile(0);
    tile.acquire(STARPU_R);
    auto tile2 = sum_ssq_tile2.get_tile(0);
    tile2.acquire(STARPU_R);
    const T *ptr = tile.get_local_ptr(), *ptr2 = tile2.get_local_ptr();
    for(Index i = 0; i < tile.nelems; i += 3)
    {
        T sum = ptr[i], sum2 = ptr2[i];
        T diff = std::abs(sum-sum2), norm = std::abs(sum);
        T threshold = 50 * norm * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            tile.release();
            tile2.release();
            std::cout << diff << " " << threshold << "\n";
            std::cout << sum << " " << sum2 << "\n";
            throw std::runtime_error("Invalid sum");
        }
        T scale = ptr[i+1], scale2 = ptr2[i+1];
        diff = std::abs(scale-scale2);
        if(diff != 0)
        {
            tile.release();
            tile2.release();
            throw std::runtime_error("Invalid scale");
        }
        T ssq = ptr[i+2], ssq2 = ptr2[i+2];
        diff = std::abs(ssq-ssq2);
        norm = std::abs(ssq);
        threshold = 50 * norm * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            tile.release();
            tile2.release();
            std::cout << diff << " " << threshold << "\n";
            std::cout << ssq << " " << ssq2 << "\n";
            throw std::runtime_error("Invalid ssq");
        }
    }
    tile.release();
    tile2.release();
}

template<typename T>
void validate_sum_ssq()
{
    Tensor<T> A({9, 10, 13, 15}, {4, 5, 6, 7}), A_tile(A.shape, A.shape);
    constexpr unsigned long long seed = 100000000000001ULL;
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    copy_intersection(A, A_tile);
    std::vector<std::vector<Index>> axes = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2},
        {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
        {1, 2, 3}, {0, 1, 2, 3}};
    for(Index i = 0; i < axes.size(); ++i)
    {
        std::vector<Index> shape(A.ndim+1-axes[i].size()), basetile(shape);
        std::vector<Index> work_shape(A.ndim+1), work_basetile(A.ndim+1);
        work_shape[0] = 3;
        work_basetile[0] = 3;
        shape[0] = 3;
        basetile[0] = 3;
        Index k = 0;
        for(Index j = 0; j < A.ndim; ++j)
        {
            if(k == axes[i].size() or axes[i][k] != j)
            {
                shape[j+1-k] = A.shape[j];
                basetile[j+1-k] = A.basetile_shape[j];
                work_shape[j+1] = A.shape[j];
                work_basetile[j+1] = A.basetile_shape[j];
            }
            else
            {
                ++k;
                work_shape[j+1] = A.grid.shape[j];
                work_basetile[j+1] = 1;
            }
        }
        Tensor<T> sum_ssq(shape, basetile),
            sum_ssq_work(work_shape, work_basetile);
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes[i]);
    }
}

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
    std::vector<std::vector<Index>> axes = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2},
        {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
        {1, 2, 3}, {0, 1, 2, 3}};
    for(Index i = 0; i < axes.size(); ++i)
    {
        std::vector<Index> shape(A.ndim+1-axes[i].size()), basetile(shape);
        std::vector<Index> work_shape(A.ndim+1), work_basetile(A.ndim+1);
        work_shape[0] = 3;
        work_basetile[0] = 3;
        shape[0] = 3;
        basetile[0] = 3;
        Index k = 0;
        Index nelems = 1;
        for(Index j = 0; j < A.ndim; ++j)
        {
            if(k == axes[i].size() or axes[i][k] != j)
            {
                shape[j+1-k] = A.shape[j];
                basetile[j+1-k] = A.basetile_shape[j];
                work_shape[j+1] = A.shape[j];
                work_basetile[j+1] = A.basetile_shape[j];
            }
            else
            {
                nelems *= A.shape[j];
                ++k;
                work_shape[j+1] = A.grid.shape[j];
                work_basetile[j+1] = 1;
            }
        }
        std::vector<Index> shape2(shape), basetile2(basetile);
        shape2[0] = 2;
        basetile2[0] = 2;
        Tensor<T> sum_ssq(shape, basetile), sum_ssq_tile(shape, shape),
            sum_ssq_work(work_shape, work_basetile);
        Tensor<T> avg_dev(shape2, basetile2), avg_dev_tile(shape2, shape2);
        norm_sum_ssq(A, sum_ssq, sum_ssq_work, axes[i]);
        copy_intersection(sum_ssq, sum_ssq_tile);
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
    //validate_avg_dev<fp64_t>();
    return 0;
}

