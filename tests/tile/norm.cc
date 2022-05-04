#include "nntile/tile/norm.hh"
#include "nntile/tile/randn.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;

template<typename T>
void check_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes)
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
        if(val == 0)
        {
            continue;
        }
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
        T norm = std::abs(local[i]);
        T threshold = std::numeric_limits<T>::epsilon() * norm;
        if(diff > threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "norm=" << norm << " diff=" << diff <<
                " threshold=" << threshold << "\n";
            delete[] local;
            sum_ssq.release();
            throw std::runtime_error("Wrong answer");
        }
    }
    delete[] local;
    sum_ssq.release();
}

template<typename T>
void check_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq, Index axis)
{
    norm_sum_ssq(src, sum_ssq, axis);
    src.acquire(STARPU_R);
    sum_ssq.acquire(STARPU_R);
    T *local = new T[sum_ssq.nelems];
    auto src_ptr = src.get_local_ptr();
    for(Index i = 0; i < sum_ssq.nelems; i += 3)
    {
        local[i] = 0; // sum
        local[i+1] = 0; // scale
        local[i+2] = 0; // scaled sum of squares
    }
    std::vector<Index> sum_ssq_index(sum_ssq.ndim);
    for(Index i = 0; i < src.nelems; ++i)
    {
        auto src_index = src.linear_to_index(i);
        for(Index j = 0; j < axis; ++j)
        {
            sum_ssq_index[j+1] = src_index[j];
        }
        for(Index j = axis+1; j < src.ndim; ++j)
        {
            sum_ssq_index[j] = src_index[j];
        }
        Index j = sum_ssq.index_to_linear(sum_ssq_index);
        T val = src_ptr[i];
        if(val == 0)
        {
            continue;
        }
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
        T norm = std::abs(local[i]);
        T threshold = 10 * std::numeric_limits<T>::epsilon() * norm;
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
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    {
        std::vector<Index> axes{0};
        Tile<T> sum_ssq({3, 5, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> sum_ssq2({3, 5, 6, 7});
        check_sum_ssq(A, sum_ssq2, axes[0]);
    }
    {
        std::vector<Index> axes{1};
        Tile<T> sum_ssq({3, 4, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> sum_ssq2({3, 4, 6, 7});
        check_sum_ssq(A, sum_ssq2, axes[0]);
    }
    {
        std::vector<Index> axes{2};
        Tile<T> sum_ssq({3, 4, 5, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> sum_ssq2({3, 4, 5, 7});
        check_sum_ssq(A, sum_ssq2, axes[0]);
    }
    {
        std::vector<Index> axes{3};
        Tile<T> sum_ssq({3, 4, 5, 6});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> sum_ssq2({3, 4, 5, 6});
        check_sum_ssq(A, sum_ssq2, axes[0]);
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
    TESTN(norm_sum_ssq(Tile<T>({4}), Tile<T>({3}), -1));
    TESTN(norm_sum_ssq(Tile<T>({4}), Tile<T>({3}), 1));
    TESTN(norm_sum_ssq(Tile<T>({3, 3}), Tile<T>({3}), 0));
    TESTN(norm_sum_ssq(Tile<T>({}), Tile<T>({}), 0));
    TESTN(norm_sum_ssq(Tile<T>({3, 3}), Tile<T>({2, 3}), 0));
    TESTN(norm_sum_ssq(Tile<T>({3, 3}), Tile<T>({3, 2}), 0));
    TESTN(norm_sum_ssq(Tile<T>({3, 3, 3}), Tile<T>({3, 2, 3}), 1));
    // Check certain predefined inputs
    T *data0 = new T[A.nelems], *data2 = new T[A.nelems];
    Tile<T> A0(static_cast<TileTraits>(A), data0, A.nelems);
    Tile<T> A2(static_cast<TileTraits>(A), data2, A.nelems);
    for(Index i = 0; i < A.nelems; ++i)
    {
        data0[i] = 0;
        data2[i] = -2;
    }
    {
        std::vector<Index> axes{0};
        const T nelems = 4, val = -8;
        Tile<T> sum_ssq0({3, 5, 6, 7}), sum_ssq2({3, 5, 6, 7});
        norm_sum_ssq(A0, sum_ssq0, axes);
        norm_sum_ssq(A2, sum_ssq2, axes);
        sum_ssq0.acquire(STARPU_R);
        sum_ssq2.acquire(STARPU_R);
        const T *ptr0 = sum_ssq0.get_local_ptr();
        const T *ptr2 = sum_ssq2.get_local_ptr();
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(ptr0[3*i] != 0 or ptr0[3*i+1]*ptr0[3*i+2] != 0)
            {
                sum_ssq0.release();
                sum_ssq2.release();
                throw std::runtime_error("0-array is not 0");
            }
            if(ptr2[3*i] != val or ptr2[3*i+1] != 2 or ptr2[3*i+2] != nelems)
            {
                sum_ssq0.release();
                sum_ssq2.release();
                throw std::runtime_error("2-array is not 2");
            }
        }
        sum_ssq0.release();
        sum_ssq2.release();
    }
    {
        std::vector<Index> axes{0};
        const T nelems = 4, val = -8;
        Tile<T> sum_ssq0({3, 5, 6, 7}), sum_ssq2({3, 5, 6, 7});
        norm_sum_ssq(A0, sum_ssq0, axes[0]);
        norm_sum_ssq(A2, sum_ssq2, axes[0]);
        sum_ssq0.acquire(STARPU_R);
        sum_ssq2.acquire(STARPU_R);
        const T *ptr0 = sum_ssq0.get_local_ptr();
        const T *ptr2 = sum_ssq2.get_local_ptr();
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(ptr0[3*i] != 0 or ptr0[3*i+1]*ptr0[3*i+2] != 0)
            {
                sum_ssq0.release();
                sum_ssq2.release();
                throw std::runtime_error("0-array is not 0");
            }
            if(ptr2[3*i] != val or ptr2[3*i+1] != 2 or ptr2[3*i+2] != nelems)
            {
                sum_ssq0.release();
                sum_ssq2.release();
                throw std::runtime_error("2-array is not 2");
            }
        }
        sum_ssq0.release();
        sum_ssq2.release();
    }
    {
        std::vector<Index> axes{1, 3};
        const T nelems = 35, val = -70;
        Tile<T> sum_ssq0({3, 4, 6}), sum_ssq2({3, 4, 6});
        norm_sum_ssq(A0, sum_ssq0, axes);
        norm_sum_ssq(A2, sum_ssq2, axes);
        sum_ssq0.acquire(STARPU_R);
        sum_ssq2.acquire(STARPU_R);
        const T *ptr0 = sum_ssq0.get_local_ptr();
        const T *ptr2 = sum_ssq2.get_local_ptr();
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(ptr0[3*i] != 0 or ptr0[3*i+1]*ptr0[3*i+2] != 0)
            {
                sum_ssq0.release();
                sum_ssq2.release();
                throw std::runtime_error("0-array is not 0");
            }
            if(ptr2[3*i] != val or ptr2[3*i+1] != 2 or ptr2[3*i+2] != nelems)
            {
                sum_ssq0.release();
                sum_ssq2.release();
                throw std::runtime_error("2-array is not 2");
            }
        }
        sum_ssq0.release();
        sum_ssq2.release();
    }
    delete[] data0;
    delete[] data2;
}

template<typename T>
void check_avg_dev(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    norm_avg_dev(sum_ssq, avg_dev, nelems, eps);
    sum_ssq.acquire(STARPU_R);
    avg_dev.acquire(STARPU_R);
    const T *sum_ssq_ptr = sum_ssq.get_local_ptr();
    const T *avg_dev_ptr = avg_dev.get_local_ptr();
    Index m = avg_dev.nelems / 2;
    for(Index i = 0; i < m; ++i)
    {
        const T &sum = sum_ssq_ptr[3*i];
        const T &scale = sum_ssq_ptr[3*i+1];
        const T &ssq = sum_ssq_ptr[3*i+2];
        const T &avg = avg_dev_ptr[2*i];
        const T &dev = avg_dev_ptr[2*i+1];
        T avg_ref = sum / nelems;
        T diff_avg = std::abs(avg - avg_ref);
        T norm_avg = std::abs(avg_ref);
        T threshold_avg = norm_avg * std::numeric_limits<T>::epsilon();
        if(diff_avg > threshold_avg)
        {
            sum_ssq.release();
            avg_dev.release();
            throw std::runtime_error("average is incorrect");
        }
        T avg_sqr = scale * scale * ssq / nelems;
        avg_sqr += eps * eps;
        T dev_ref = std::sqrt(avg_sqr - avg_ref*avg_ref);
        T diff_dev = std::abs(dev_ref - dev);
        T threshold_dev = (dev_ref) * std::numeric_limits<T>::epsilon();
        // If avg_sqr is close to avg_ref^2 then threshold must be updated
        threshold_dev *= 2 + avg_ref*avg_ref/dev_ref/dev_ref;
        if(diff_dev > threshold_dev)
        {
            std::cerr << "dev=" << dev << " dev_ref=" << dev_ref << "\n";
            std::cerr << "diff=" << diff_dev << " threshold=" << threshold_dev
                << "\n";
            std::cerr << "sum=" << sum << " scale=" << scale << " ssq=" << ssq
                << " nelems=" << nelems << "\n";
            sum_ssq.release();
            avg_dev.release();
            throw std::runtime_error("deviation is incorrect");
        }
    }
    sum_ssq.release();
    avg_dev.release();
}

template<typename T>
void validate_avg_dev()
{
    Tile<T> A({4, 5, 6, 7}); 
    constexpr unsigned long long seed = 100000000000001ULL;
    constexpr T eps0 = 0, eps1 = 0.01, eps2=1e+10;
    randn(A, seed);
    {
        std::vector<Index> axes{0};
        Tile<T> sum_ssq({3, 5, 6, 7});
        norm_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 5, 6, 7});
        check_avg_dev(sum_ssq, avg_dev, 4, eps0);
        check_avg_dev(sum_ssq, avg_dev, 4, eps1);
        check_avg_dev(sum_ssq, avg_dev, 4, eps2);
    }
    {
        std::vector<Index> axes{1};
        Tile<T> sum_ssq({3, 4, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4, 6, 7});
        check_avg_dev(sum_ssq, avg_dev, 5, eps0);
        check_avg_dev(sum_ssq, avg_dev, 5, eps1);
        check_avg_dev(sum_ssq, avg_dev, 5, eps2);
    }
    {
        std::vector<Index> axes{2};
        Tile<T> sum_ssq({3, 4, 5, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4, 5, 7});
        check_avg_dev(sum_ssq, avg_dev, 6, eps0);
        check_avg_dev(sum_ssq, avg_dev, 6, eps1);
        check_avg_dev(sum_ssq, avg_dev, 6, eps2);
    }
    {
        std::vector<Index> axes{3};
        Tile<T> sum_ssq({3, 4, 5, 6});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4, 5, 6});
        check_avg_dev(sum_ssq, avg_dev, 7, eps0);
        check_avg_dev(sum_ssq, avg_dev, 7, eps1);
        check_avg_dev(sum_ssq, avg_dev, 7, eps2);
    }
    {
        std::vector<Index> axes{0, 1};
        Tile<T> sum_ssq({3, 6, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 6, 7});
        check_avg_dev(sum_ssq, avg_dev, 20, eps0);
        check_avg_dev(sum_ssq, avg_dev, 20, eps1);
        check_avg_dev(sum_ssq, avg_dev, 20, eps2);
    }
    {
        std::vector<Index> axes{0, 2};
        Tile<T> sum_ssq({3, 5, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 5, 7});
        check_avg_dev(sum_ssq, avg_dev, 24, eps0);
        check_avg_dev(sum_ssq, avg_dev, 24, eps1);
        check_avg_dev(sum_ssq, avg_dev, 24, eps2);
    }
    {
        std::vector<Index> axes{0, 3};
        Tile<T> sum_ssq({3, 5, 6});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 5, 6});
        check_avg_dev(sum_ssq, avg_dev, 28, eps0);
        check_avg_dev(sum_ssq, avg_dev, 28, eps1);
        check_avg_dev(sum_ssq, avg_dev, 28, eps2);
    }
    {
        std::vector<Index> axes{1, 2};
        Tile<T> sum_ssq({3, 4, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4, 7});
        check_avg_dev(sum_ssq, avg_dev, 30, eps0);
        check_avg_dev(sum_ssq, avg_dev, 30, eps1);
        check_avg_dev(sum_ssq, avg_dev, 30, eps2);
    }
    {
        std::vector<Index> axes{1, 3};
        Tile<T> sum_ssq({3, 4, 6});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4, 6});
        check_avg_dev(sum_ssq, avg_dev, 35, eps0);
        check_avg_dev(sum_ssq, avg_dev, 35, eps1);
        check_avg_dev(sum_ssq, avg_dev, 35, eps2);
    }
    {
        std::vector<Index> axes{2, 3};
        Tile<T> sum_ssq({3, 4, 5});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4, 5});
        check_avg_dev(sum_ssq, avg_dev, 42, eps0);
        check_avg_dev(sum_ssq, avg_dev, 42, eps1);
        check_avg_dev(sum_ssq, avg_dev, 42, eps2);
    }
    {
        std::vector<Index> axes{0, 1, 2};
        Tile<T> sum_ssq({3, 7});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 7});
        check_avg_dev(sum_ssq, avg_dev, 120, eps0);
        check_avg_dev(sum_ssq, avg_dev, 120, eps1);
        check_avg_dev(sum_ssq, avg_dev, 120, eps2);
    }
    {
        std::vector<Index> axes{0, 1, 3};
        Tile<T> sum_ssq({3, 6});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 6});
        check_avg_dev(sum_ssq, avg_dev, 140, eps0);
        check_avg_dev(sum_ssq, avg_dev, 140, eps1);
        check_avg_dev(sum_ssq, avg_dev, 140, eps2);
    }
    {
        std::vector<Index> axes{0, 2, 3};
        Tile<T> sum_ssq({3, 5});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 5});
        check_avg_dev(sum_ssq, avg_dev, 168, eps0);
        check_avg_dev(sum_ssq, avg_dev, 168, eps1);
        check_avg_dev(sum_ssq, avg_dev, 168, eps2);
    }
    {
        std::vector<Index> axes{1, 2, 3};
        Tile<T> sum_ssq({3, 4});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2, 4});
        check_avg_dev(sum_ssq, avg_dev, 210, eps0);
        check_avg_dev(sum_ssq, avg_dev, 210, eps1);
        check_avg_dev(sum_ssq, avg_dev, 210, eps2);
    }
    {
        std::vector<Index> axes{0, 1, 2, 3};
        Tile<T> sum_ssq({3});
        check_sum_ssq(A, sum_ssq, axes);
        Tile<T> avg_dev({2});
        check_avg_dev(sum_ssq, avg_dev, 840, eps0);
        check_avg_dev(sum_ssq, avg_dev, 840, eps1);
        check_avg_dev(sum_ssq, avg_dev, 840, eps2);
    }
    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2}), -1, T{0}));
    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2}), 0, T{0}));
    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2}), 1, T{-1}));
    TESTN(norm_avg_dev(Tile<T>({2}), Tile<T>({2}), 1, T{1}));
    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({3}), 1, T{1}));
    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2, 3}), 1, T{1}));
    TESTN(norm_avg_dev(Tile<T>({}), Tile<T>({}), 1, T{1}));
    TESTN(norm_avg_dev(Tile<T>({3, 4}), Tile<T>({2, 3}), 1, T{1}));
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

