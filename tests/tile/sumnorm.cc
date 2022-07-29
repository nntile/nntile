#include "nntile/tile/clear.hh"
#include "nntile/tile/sumnorm.hh"
#include "nntile/tile/randn.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;

Starpu starpu;

template<typename T>
void check_sumnorm(const Tile<T> &src, const Tile<T> &dst, Index axis)
{
    clear_async(dst);
    sumnorm(src, dst, axis);
    Starpu::pause();
    auto src_local = src.acquire(STARPU_R), dst_local = dst.acquire(STARPU_RW);
    std::vector<T> local(dst.nelems);
    for(Index i = 0; i < dst.nelems; i += 2)
    {
        local[i] = 0; // sum
        local[i+1] = 0; // sum of squares
    }
    std::vector<Index> dst_index(dst.ndim);
    for(Index i = 0; i < src.nelems; ++i)
    {
        auto src_index = src.linear_to_index(i);
        for(Index j = 0; j < axis; ++j)
        {
            dst_index[j+1] = src_index[j];
        }
        for(Index j = axis+1; j < src.ndim; ++j)
        {
            dst_index[j] = src_index[j];
        }
        Index j = dst.index_to_linear(dst_index);
        T val = src_local[i];
        if(val == 0)
        {
            continue;
        }
        local[j] += val;
        local[j+1] = std::sqrt(local[j+1]*local[j+1] + val*val);
    }
    src_local.release();
    for(Index i = 0; i < dst.nelems; i += 2)
    {
        T sum_diff = std::abs(local[i] - dst_local[i]);
        T sum_ref = std::abs(local[i]);
        T sum_threshold = 10 * std::numeric_limits<T>::epsilon() * sum_ref;
        if(sum_diff > sum_threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "sum_ref=" << sum_ref << " sum_diff=" << sum_diff
                << " sum_threshold=" << sum_threshold << "\n";
            throw std::runtime_error("Wrong sum");
        }
        T norm_diff = std::abs(local[i+1] - dst_local[i+1]);
        T norm_ref = std::abs(local[i+1]);
        T norm_threshold = 10 * std::numeric_limits<T>::epsilon() * norm_ref;
        if(norm_diff > norm_threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "norm_ref=" << norm_ref << " norm_diff=" << norm_diff
                << " norm_threshold=" << norm_threshold << "\n";
            throw std::runtime_error("Wrong norm");
        }
    }
    for(Index i = 0; i < dst.nelems; i += 2)
    {
        dst_local[i] *= 0.1;
        dst_local[i+1] *= 0.1;
    }
    dst_local.release();
    // All the local handles/buffers shall be released before submitting
    // blocking codelets
    Starpu::resume();
    sumnorm(src, dst, axis);
    Starpu::pause();
    dst_local.acquire(STARPU_R);
    for(Index i = 0; i < dst.nelems; i += 2)
    {
        T sum_diff = std::abs(1.1*local[i] - dst_local[i]);
        T sum_norm = std::abs(1.1*local[i]);
        T sum_threshold = 10 * std::numeric_limits<T>::epsilon() * sum_norm;
        if(sum_diff > sum_threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "sum_norm=" << sum_norm << " sum_diff=" << sum_diff <<
                " sum_threshold=" << sum_threshold << "\n";
            throw std::runtime_error("Wrong sum");
        }
        constexpr T f = std::sqrt(1.01);
        T norm_diff = std::abs(f*local[i+1] - dst_local[i+1]);
        T norm_ref = std::abs(f*local[i+1]);
        T norm_threshold = 10 * std::numeric_limits<T>::epsilon() * norm_ref;
        if(norm_diff > norm_threshold)
        {
            std::cout << "i=" << i << "\n";
            std::cout << "norm_ref=" << norm_ref << " norm_diff=" << norm_diff <<
                " norm_threshold=" << norm_threshold << "\n";
            throw std::runtime_error("Wrong norm");
        }
    }
    Starpu::resume();
}

template<typename T>
void validate_sumnorm()
{
    Tile<T> A({4, 5, 6, 7}); 
    constexpr unsigned long long seed = 100000000000001ULL;
    // Avoid mean=0 because of instable relative error of sum (division by 0)
    randn(A, seed, T{1}, T{1});
    for(Index i = 0; i < A.ndim; ++i)
    {
        std::vector<Index> shape(A.ndim);
        shape[0] = 2;
        for(Index j = 0; j < i; ++j)
        {
            shape[j+1] = A.shape[j];
        }
        for(Index j = i+1; j < A.ndim; ++j)
        {
            shape[j] = A.shape[j];
        }
        Tile<T> dst(shape);
        check_sumnorm(A, dst, i);
        for(Index j = 1; j < shape.size(); ++j)
        {
            auto shape2(shape);
            ++shape2[j];
            TESTN(sumnorm(A, Tile<T>(shape2), i));
        }
    }
    TESTN(sumnorm(Tile<T>({4}), Tile<T>({3}), -1));
    TESTN(sumnorm(Tile<T>({4}), Tile<T>({3}), 1));
    TESTN(sumnorm(Tile<T>({3, 3}), Tile<T>({3}), 0));
    TESTN(sumnorm(Tile<T>({}), Tile<T>({}), 0));
    TESTN(sumnorm(Tile<T>({3, 3}), Tile<T>({3, 3}), 0));
    TESTN(sumnorm(Tile<T>({3, 3}), Tile<T>({3, 2}), 0));
    TESTN(sumnorm(Tile<T>({3, 3, 3}), Tile<T>({3, 2, 3}), 1));
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
        Index axis{0};
        const T nelems = 4, val = -8;
        Tile<T> dst0({2, 5, 6, 7}), dst2({2, 5, 6, 7});
        clear_async(dst0);
        clear_async(dst2);
        sumnorm(A0, dst0, axis);
        sumnorm(A2, dst2, axis);
        auto local0 = dst0.acquire(STARPU_R),
             local2 = dst2.acquire(STARPU_R);
        for(Index i = 0; i < dst0.nelems/2; ++i)
        {
            if(local0[2*i] != 0 or local0[2*i+1] != 0)
            {
                throw std::runtime_error("0-array is not 0");
            }
            if(local2[2*i] != val or local2[2*i+1] != 2*std::sqrt(nelems))
            {
                throw std::runtime_error("2-array is not 2");
            }
        }
    }
}

int main(int argc, char **argv)
{
    validate_sumnorm<fp32_t>();
    validate_sumnorm<fp64_t>();
    return 0;
}

