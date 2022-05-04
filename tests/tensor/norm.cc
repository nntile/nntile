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
    {
        std::vector<Index> axes{0};
        Tensor<T> sum_ssq({3, 10, 13, 15}, {3, 5, 6, 7}),
            sum_ssq_work({3, 3, 10, 13, 15}, {3, 1, 5, 6, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{1};
        Tensor<T> sum_ssq({3, 9, 13, 15}, {3, 4, 6, 7}),
            sum_ssq_work({3, 9, 2, 13, 15}, {3, 4, 1, 6, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{2};
        Tensor<T> sum_ssq({3, 9, 10, 15}, {3, 4, 5, 7}),
            sum_ssq_work({3, 9, 10, 3, 15}, {3, 4, 5, 1, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{3};
        Tensor<T> sum_ssq({3, 9, 10, 13}, {3, 4, 5, 6}),
            sum_ssq_work({3, 9, 10, 13, 3}, {3, 4, 5, 6, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 1};
        Tensor<T> sum_ssq({3, 13, 15}, {3, 6, 7}),
            sum_ssq_work({3, 3, 2, 13, 15}, {3, 1, 1, 6, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 2};
        Tensor<T> sum_ssq({3, 10, 15}, {3, 5, 7}),
            sum_ssq_work({3, 3, 10, 3, 15}, {3, 1, 5, 1, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 3};
        Tensor<T> sum_ssq({3, 10, 13}, {3, 5, 6}),
            sum_ssq_work({3, 3, 10, 13, 3}, {3, 1, 5, 6, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{1, 2};
        Tensor<T> sum_ssq({3, 9, 15}, {3, 4, 7}),
            sum_ssq_work({3, 9, 2, 3, 15}, {3, 4, 1, 1, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{1, 3};
        Tensor<T> sum_ssq({3, 9, 13}, {3, 4, 6}),
            sum_ssq_work({3, 9, 2, 13, 3}, {3, 4, 1, 6, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{2, 3};
        Tensor<T> sum_ssq({3, 9, 10}, {3, 4, 5}),
            sum_ssq_work({3, 9, 10, 3, 3}, {3, 4, 5, 1, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 1, 2};
        Tensor<T> sum_ssq({3, 15}, {3, 7}),
            sum_ssq_work({3, 3, 2, 3, 15}, {3, 1, 1, 1, 7});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 1, 3};
        Tensor<T> sum_ssq({3, 13}, {3, 6}),
            sum_ssq_work({3, 3, 2, 13, 3}, {3, 1, 1, 6, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 2, 3};
        Tensor<T> sum_ssq({3, 10}, {3, 5}),
            sum_ssq_work({3, 3, 10, 3, 3}, {3, 1, 5, 1, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{1, 2, 3};
        Tensor<T> sum_ssq({3, 9}, {3, 4}),
            sum_ssq_work({3, 9, 2, 3, 3}, {3, 4, 1, 1, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
    {
        std::vector<Index> axes{0, 1, 2, 3};
        Tensor<T> sum_ssq({3}, {3}),
            sum_ssq_work({3, 3, 2, 3, 3}, {3, 1, 1, 1, 1});
        check_sum_ssq(A, A_tile, sum_ssq, sum_ssq_work, axes);
    }
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_sum_ssq<fp32_t>();
    validate_sum_ssq<fp64_t>();
    return 0;
}

