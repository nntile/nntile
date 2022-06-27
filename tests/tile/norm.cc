#include "nntile/tile/norm.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;

template<typename T>
void check_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq,
        const std::vector<Index> &axes)
{
    norm_sum_ssq(src, sum_ssq, axes, true);
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
    auto src_local = src.acquire(STARPU_R),
         sum_ssq_local = sum_ssq.acquire(STARPU_RW);
    std::vector<T> local(sum_ssq.nelems);
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
        T val = src_local[i];
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
    src_local.release();
    for(Index i = 0; i < sum_ssq.nelems; ++i)
    {
        T diff = std::abs(local[i] - sum_ssq_local[i]);
        T norm = std::abs(local[i]);
        T threshold = std::numeric_limits<T>::epsilon() * norm;
        if(diff > threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "norm=" << norm << " diff=" << diff <<
                " threshold=" << threshold << "\n";
            throw std::runtime_error("Wrong answer");
        }
    }
    for(Index i = 0; i < sum_ssq.nelems; i += 3)
    {
        sum_ssq_local[i] *= 0.1;
        sum_ssq_local[i+1] *= 0.1;
    }
    sum_ssq_local.release();
    // All the local handles/buffers shall be released before submitting
    // blocking codelets
    norm_sum_ssq(src, sum_ssq, axes, false);
    sum_ssq_local.acquire(STARPU_R);
    for(Index i = 0; i < sum_ssq.nelems; i += 3)
    {
        T sum_ref = 1.1 * local[i];
        T sum_norm = std::abs(sum_ref);
        T sum_diff = std::abs(sum_ref - sum_ssq_local[i]);
        T sum_threshold = 16 * sum_norm * std::numeric_limits<T>::epsilon();
        if(sum_diff > sum_threshold)
        {
            std::cout << sum_diff << " " << sum_threshold << "\n";
            std::cout << sum_ref << " " << sum_ssq_local[i] << "\n";
            throw std::runtime_error("Error in sum");
        }
        T scale_ref = local[i+1];
        T scale_norm = scale_ref;
        T scale_diff = std::abs(scale_ref - sum_ssq_local[i+1]);
        T scale_threshold = scale_norm * std::numeric_limits<T>::epsilon();
        if(scale_diff > scale_threshold)
        {
            std::cout << scale_diff << " " << scale_threshold << "\n";
            std::cout << scale_ref << " " << sum_ssq_local[i+1] << "\n";
            throw std::runtime_error("Error in scale");
        }
        T ssq_ref = 1.01 * local[i+2];
        T ssq_norm = ssq_ref;
        T ssq_diff = std::abs(ssq_ref - sum_ssq_local[i+2]);
        T ssq_threshold = 8 * ssq_norm * std::numeric_limits<T>::epsilon();
        if(ssq_diff > ssq_threshold)
        {
            std::cout << ssq_diff << " " << ssq_threshold << "\n";
            std::cout << ssq_ref << " " << sum_ssq_local[i+2] << "\n";
            throw std::runtime_error("Error in ssq");
        }
    }
}

template<typename T>
void check_sum_ssq(const Tile<T> &src, const Tile<T> &sum_ssq, Index axis)
{
    norm_sum_ssq(src, sum_ssq, axis, true);
    auto src_local = src.acquire(STARPU_R),
         sum_ssq_local = sum_ssq.acquire(STARPU_RW);
    std::vector<T> local(sum_ssq.nelems);
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
        T val = src_local[i];
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
    src_local.release();
    for(Index i = 0; i < sum_ssq.nelems; ++i)
    {
        T diff = std::abs(local[i] - sum_ssq_local[i]);
        T norm = std::abs(local[i]);
        T threshold = 10 * std::numeric_limits<T>::epsilon() * norm;
        if(diff > threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "norm=" << norm << " diff=" << diff << " threshold="
                << threshold << "\n";
            throw std::runtime_error("Wrong answer");
        }
    }
    for(Index i = 0; i < sum_ssq.nelems; i += 3)
    {
        sum_ssq_local[i] *= 0.1;
        sum_ssq_local[i+1] *= 0.1;
    }
    sum_ssq_local.release();
    // All the local handles/buffers shall be released before submitting
    // blocking codelets
    norm_sum_ssq(src, sum_ssq, axis, false);
    sum_ssq_local.acquire(STARPU_R);
    for(Index i = 0; i < sum_ssq.nelems; i += 3)
    {
        T sum_diff = std::abs(1.1*local[i] - sum_ssq_local[i]);
        T sum_norm = std::abs(1.1*local[i]);
        T sum_threshold = 10 * std::numeric_limits<T>::epsilon() * sum_norm;
        if(sum_diff > sum_threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "sum_norm=" << sum_norm << " sum_diff=" << sum_diff <<
                " sum_threshold=" << sum_threshold << "\n";
            throw std::runtime_error("Wrong sum");
        }
    }
}

template<typename T>
void validate_sum_ssq()
{
    Tile<T> A({4, 5, 6, 7}); 
    constexpr unsigned long long seed = 100000000000001ULL;
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    std::vector<std::vector<Index>> axes = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2},
        {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
        {1, 2, 3}, {0, 1, 2, 3}};
    for(Index i = 0; i < axes.size(); ++i)
    {
        std::vector<Index> shape(A.ndim+1-axes[i].size());
        shape[0] = 3;
        Index k = 0;
        for(Index j = 0; j < A.ndim; ++j)
        {
            if(k == axes[i].size() or axes[i][k] != j)
            {
                shape[j+1-k] = A.shape[j];
            }
            else
            {
                ++k;
            }
        }
        Tile<T> sum_ssq(shape);
        check_sum_ssq(A, sum_ssq, axes[i]);
        for(Index j = 1; j < shape.size(); ++j)
        {
            auto shape2(shape);
            ++shape2[j];
            TESTN(norm_sum_ssq(A, Tile<T>(shape2), axes[i]));
        }
    }
    for(Index i = 0; i < A.ndim; ++i)
    {
        std::vector<Index> shape(A.ndim);
        shape[0] = 3;
        for(Index j = 0; j < i; ++j)
        {
            shape[j+1] = A.shape[j];
        }
        for(Index j = i+1; j < A.ndim; ++j)
        {
            shape[j] = A.shape[j];
        }
        Tile<T> sum_ssq(shape);
        check_sum_ssq(A, sum_ssq, i);
        for(Index j = 1; j < shape.size(); ++j)
        {
            auto shape2(shape);
            ++shape2[j];
            TESTN(norm_sum_ssq(A, Tile<T>(shape2), i));
        }
    }
    return;
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
    std::vector<T> data0(A.nelems), data2(A.nelems);
    for(Index i = 0; i < A.nelems; ++i)
    {
        data0[i] = 0;
        data2[i] = -2;
    }
    Tile<T> A0(static_cast<TileTraits>(A), &data0[0], A.nelems);
    Tile<T> A2(static_cast<TileTraits>(A), &data2[0], A.nelems);
    {
        std::vector<Index> axes{0};
        const T nelems = 4, val = -8;
        Tile<T> sum_ssq0({3, 5, 6, 7}), sum_ssq2({3, 5, 6, 7});
        norm_sum_ssq(A0, sum_ssq0, axes, true);
        norm_sum_ssq(A2, sum_ssq2, axes, true);
        auto local0 = sum_ssq0.acquire(STARPU_R),
             local2 = sum_ssq2.acquire(STARPU_R);
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(local0[3*i] != 0 or local0[3*i+1]*local0[3*i+2] != 0)
            {
                throw std::runtime_error("0-array is not 0");
            }
            if(local2[3*i] != val or local2[3*i+1] != 2 or
                    local2[3*i+2] != nelems)
            {
                throw std::runtime_error("2-array is not 2");
            }
        }
        local0.release();
        local2.release();
        // All the local handles/buffers shall be released before submitting
        // blocking codelets
        norm_sum_ssq(A2, sum_ssq0, axes, false);
        local0.acquire(STARPU_R);
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(local0[3*i] != val or local0[3*i+1] != 2 or
                    local0[3*i+2] != nelems)
            {
                throw std::runtime_error("2-array is not 2");
            }
        }
        local0.release();
        // All the local handles/buffers shall be released before submitting
        // blocking codelets
        norm_sum_ssq(A0, sum_ssq0, axes, false);
        local0.acquire(STARPU_R);
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(local0[3*i] != val or local0[3*i+1] != 2 or
                    local0[3*i+2] != nelems)
            {
                throw std::runtime_error("2-array is not 2");
            }
        }
    }
    {
        std::vector<Index> axes{0};
        const T nelems = 4, val = -8;
        Tile<T> sum_ssq0({3, 5, 6, 7}), sum_ssq2({3, 5, 6, 7});
        norm_sum_ssq(A0, sum_ssq0, axes[0]);
        norm_sum_ssq(A2, sum_ssq2, axes[0]);
        auto local0 = sum_ssq0.acquire(STARPU_R),
             local2 = sum_ssq2.acquire(STARPU_R);
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(local0[3*i] != 0 or local0[3*i+1]*local0[3*i+2] != 0)
            {
                throw std::runtime_error("0-array is not 0");
            }
            if(local2[3*i] != val or local2[3*i+1] != 2 or
                    local2[3*i+2] != nelems)
            {
                throw std::runtime_error("2-array is not 2");
            }
        }
    }
    {
        std::vector<Index> axes{1, 3};
        const T nelems = 35, val = -70;
        Tile<T> sum_ssq0({3, 4, 6}), sum_ssq2({3, 4, 6});
        norm_sum_ssq(A0, sum_ssq0, axes);
        norm_sum_ssq(A2, sum_ssq2, axes);
        auto local0 = sum_ssq0.acquire(STARPU_R),
             local2 = sum_ssq2.acquire(STARPU_R);
        for(Index i = 0; i < sum_ssq0.nelems/3; ++i)
        {
            if(local0[3*i] != 0 or local0[3*i+1]*local0[3*i+2] != 0)
            {
                throw std::runtime_error("0-array is not 0");
            }
            if(local2[3*i] != val or local2[3*i+1] != 2 or
                    local2[3*i+2] != nelems)
            {
                throw std::runtime_error("2-array is not 2");
            }
        }
    }
}

template<typename T>
void check_avg_dev(const Tile<T> &sum_ssq, const Tile<T> &avg_dev,
        Index nelems, T eps)
{
    norm_avg_dev(sum_ssq, avg_dev, nelems, eps);
    auto sum_ssq_local = sum_ssq.acquire(STARPU_R),
         avg_dev_local = avg_dev.acquire(STARPU_R);
    Index m = avg_dev.nelems / 2;
    for(Index i = 0; i < m; ++i)
    {
        const T &sum = sum_ssq_local[3*i];
        const T &scale = sum_ssq_local[3*i+1];
        const T &ssq = sum_ssq_local[3*i+2];
        const T &avg = avg_dev_local[2*i];
        const T &dev = avg_dev_local[2*i+1];
        T avg_ref = sum / nelems;
        T diff_avg = std::abs(avg - avg_ref);
        T norm_avg = std::abs(avg_ref);
        T threshold_avg = norm_avg * std::numeric_limits<T>::epsilon();
        if(diff_avg > threshold_avg)
        {
            std::cerr << "diff=" << diff_avg << " threshold=" << threshold_avg
                << "\n";
            throw std::runtime_error("average is incorrect");
        }
        T avg_sqr = scale * scale * ssq / nelems;
        avg_sqr += eps * eps;
        T dev_ref = std::sqrt(avg_sqr - avg_ref*avg_ref);
        T diff_dev = std::abs(dev_ref - dev);
        T threshold_dev = (dev_ref) * std::numeric_limits<T>::epsilon();
        // If avg_sqr is close to avg_ref^2 then threshold must be updated
        threshold_dev *= 2 + 2*avg_sqr/dev_ref/dev_ref;
        if(diff_dev > threshold_dev)
        {
            std::cerr << "dev=" << dev << " dev_ref=" << dev_ref << "\n";
            std::cerr << "diff=" << diff_dev << " threshold=" << threshold_dev
                << "\n";
            std::cerr << "sum=" << sum << " scale=" << scale << " ssq=" << ssq
                << " nelems=" << nelems << "\n";
            throw std::runtime_error("deviation is incorrect");
        }
    }
}

template<typename T>
void validate_avg_dev()
{
    Tile<T> A({4, 5, 6, 7}); 
    constexpr unsigned long long seed = 100000000000001ULL;
    constexpr T eps0 = 0, eps1 = 0.01, eps2=1e+10;
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    std::vector<std::vector<Index>> axes = {{0}, {1}, {2}, {3}, {0, 1}, {0, 2},
        {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3},
        {1, 2, 3}, {0, 1, 2, 3}};
    for(Index i = 0; i < axes.size(); ++i)
    {
        std::vector<Index> shape(A.ndim+1-axes[i].size());
        std::vector<Index> shape2(A.ndim+1-axes[i].size());
        shape[0] = 3;
        shape2[0] = 2;
        Index k = 0;
        Index nelems = 1;
        for(Index j = 0; j < A.ndim; ++j)
        {
            if(k == axes[i].size() or axes[i][k] != j)
            {
                shape[j+1-k] = A.shape[j];
                shape2[j+1-k] = A.shape[j];
            }
            else
            {
                nelems *= A.shape[j];
                ++k;
            }
        }
        Tile<T> sum_ssq(shape), avg_dev(shape2);
        norm_sum_ssq(A, sum_ssq, axes[i]);
        check_avg_dev(sum_ssq, avg_dev, 1, eps0);
        check_avg_dev(sum_ssq, avg_dev, nelems, eps0);
        check_avg_dev(sum_ssq, avg_dev, nelems, eps1);
        check_avg_dev(sum_ssq, avg_dev, nelems, eps2);
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

