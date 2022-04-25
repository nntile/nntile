#include "nntile/tile/norm.hh"
#include "nntile/tile/randn.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> axes)
{
    norm_sum_ssq(src, sum_ssq, axes);
    std::vector<Index> sum_ssq_axes(sum_ssq.ndim-1);
    Index nchecked_axes = 0;
    for(Index i = 0; i < src.ndim; ++i)
    {
        if(nchecked_axes < axes.size() and axes[nchecked_axes] == i)
        {
            ++nchecked_axes;
        }
        else
        {
            sum_ssq_axes[i-nchecked_axes] = i;
        }
    }
    src.acquire(STARPU_R);
    sum_ssq.acquire(STARPU_R);
    T *local = new T[sum_ssq.nelems];
    auto src_ptr = src.get_local_ptr();
    for(Index i = 0; i < sum_ssq.nelems; i += 3)
    {
        local[i] = 0; // sum
        local[i+1] = 0; // scale
        local[i+2] = 1; // scaled sum of squares
    }
    for(Index i = 0; i < src.nelems; ++i)
    {
        auto src_index = src.linear_to_index(i);
        std::vector<Index> sum_ssq_index(sum_ssq.ndim);
        for(Index j = 1; j < sum_ssq.ndim; ++j)
        {
            sum_ssq_index[j] = src_index[sum_ssq_axes[j-1]];
        }
        Index j = sum_ssq.index_to_linear(sum_ssq_index);
        T val = src_ptr[i];
        local[j] += val;
        T absval = std::abs(val);
        if(absval > local[j+1])
        {
            T tmp = local[j+1] / absval;
            local[j+1] = absval;
            local[j+2] = local[j+2]*tmp*tmp + 1;
        }
        else
        {
            T tmp = absval / local[j+1];
            local[j+2] += tmp * tmp;
        }
    }
    src.release();
    auto sum_ssq_ptr = sum_ssq.get_local_ptr();
    for(Index i = 0; i < sum_ssq.nelems; ++i)
    {
        T diff = std::abs(local[i] - sum_ssq_ptr[i]);
        T norm = std::abs(sum_ssq_ptr[i]);
        T threshold = std::numeric_limits<T>::epsilon() * norm;
        if(diff > threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "norm=" << norm << " diff=" << diff << " threshold="
                << threshold << "\n";
            delete[] local;
            sum_ssq.release();
            throw std::runtime_error("Wrong answer");
        }
    }
    delete[] local;
    sum_ssq.release();
}

template<typename T>
void validate_sum_ssq()
{
    Tile<T> A({4, 5, 6, 7}); 
    constexpr unsigned long long seed = 100000000000001ULL;
    randn(A, seed);
    {
        std::vector<Index> axes{0};
        Tile<T> sum_ssq({3, 5, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{1};
        Tile<T> sum_ssq({3, 4, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{2};
        Tile<T> sum_ssq({3, 4, 5, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{3};
        Tile<T> sum_ssq({3, 4, 5, 6});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 1};
        Tile<T> sum_ssq({3, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 2};
        Tile<T> sum_ssq({3, 5, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 3};
        Tile<T> sum_ssq({3, 5, 6});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{1, 2};
        Tile<T> sum_ssq({3, 4, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{1, 3};
        Tile<T> sum_ssq({3, 4, 6});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{2, 3};
        Tile<T> sum_ssq({3, 4, 5});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 1, 2};
        Tile<T> sum_ssq({3, 7});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 1, 3};
        Tile<T> sum_ssq({3, 6});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 2, 3};
        Tile<T> sum_ssq({3, 5});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{1, 2, 3};
        Tile<T> sum_ssq({3, 4});
        check_sum_ssq(A, sum_ssq, axes);
    }
    {
        std::vector<Index> axes{0, 1, 2, 3};
        Tile<T> sum_ssq({3});
        check_sum_ssq(A, sum_ssq, axes);
    }
    TESTN(norm_sum_ssq(A, Tile<T>({2, 6}), std::vector<Index>{0, 1, 3}));
    TESTN(norm_sum_ssq(A, Tile<T>({3, 5}), std::vector<Index>{0, 1, 3}));
    TESTN(norm_sum_ssq(A, Tile<T>({3, 4, 6}), std::vector<Index>{0, 1, 3}));
    TESTN(norm_sum_ssq(A, Tile<T>({3}), std::vector<Index>{0, 1, 3, 2}));
    TESTN(norm_sum_ssq(Tile<T>({}), Tile<T>({3}), std::vector<Index>{}));
    TESTN(norm_sum_ssq(Tile<T>({4}), Tile<T>({3, 4}), std::vector<Index>{}));
    TESTN(norm_sum_ssq(Tile<T>({4}), Tile<T>({3}), std::vector<Index>{1}));
    TESTN(norm_sum_ssq(Tile<T>({4}), Tile<T>({3}), std::vector<Index>{-1}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_sum_ssq<fp32_t>();
    validate_sum_ssq<fp64_t>();
    return 0;
}

