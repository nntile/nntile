#include "nntile/tile/clear.hh"
#include "nntile/tile/sumnorm.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"
#include "../testing.hh"
#include <cmath>

using namespace nntile;

Starpu starpu;

//template<typename T>
//void check_avg_dev(const Tile<T> &sumnorm, const Tile<T> &avg_dev,
//        Index nelems, T eps)
//{
//    norm_avg_dev(sumnorm, avg_dev, nelems, eps);
//    auto sumnorm_local = sumnorm.acquire(STARPU_R),
//         avg_dev_local = avg_dev.acquire(STARPU_R);
//    Index m = avg_dev.nelems / 2;
//    for(Index i = 0; i < m; ++i)
//    {
//        const T &sum = sumnorm_local[3*i];
//        const T &scale = sumnorm_local[3*i+1];
//        const T &ssq = sumnorm_local[3*i+2];
//        const T &avg = avg_dev_local[2*i];
//        const T &dev = avg_dev_local[2*i+1];
//        T avg_ref = sum / nelems;
//        T diff_avg = std::abs(avg - avg_ref);
//        T norm_avg = std::abs(avg_ref);
//        T threshold_avg = norm_avg * std::numeric_limits<T>::epsilon();
//        if(diff_avg > threshold_avg)
//        {
//            std::cerr << "diff=" << diff_avg << " threshold=" << threshold_avg
//                << "\n";
//            throw std::runtime_error("average is incorrect");
//        }
//        T avg_sqr = scale * scale * ssq / nelems;
//        avg_sqr += eps * eps;
//        T dev_ref = std::sqrt(avg_sqr - avg_ref*avg_ref);
//        T diff_dev = std::abs(dev_ref - dev);
//        T threshold_dev = (dev_ref) * std::numeric_limits<T>::epsilon();
//        // If avg_sqr is close to avg_ref^2 then threshold must be updated
//        threshold_dev *= 2 + 2*avg_sqr/dev_ref/dev_ref;
//        if(diff_dev > threshold_dev)
//        {
//            std::cerr << "dev=" << dev << " dev_ref=" << dev_ref << "\n";
//            std::cerr << "diff=" << diff_dev << " threshold=" << threshold_dev
//                << "\n";
//            std::cerr << "sum=" << sum << " scale=" << scale << " ssq=" << ssq
//                << " nelems=" << nelems << "\n";
//            throw std::runtime_error("deviation is incorrect");
//        }
//    }
//}
//
//template<typename T>
//void validate_avg_dev()
//{
//    Tile<T> A({4, 5, 6, 7}); 
//    constexpr unsigned long long seed = 100000000000001ULL;
//    constexpr T eps0 = 0, eps1 = 0.01, eps2=1e+10;
//    // Avoid mean=0 because of instable relative error of sum (division by 0)
//    randn(A, seed, T{1}, T{1});
//    for(Index i = 0; i < A.ndim; ++i)
//    {
//        std::vector<Index> shape(A.ndim), shape2(A.ndim);
//        shape[0] = 3;
//        shape2[0] = 2;
//        Index k = 0;
//        Index nelems = A.shape[i];
//        for(Index j = 0; j < i; ++j)
//        {
//            shape[j+1] = A.shape[j];
//            shape2[j+1] = A.shape[j];
//        }
//        for(Index j = i+1; j < A.ndim; ++j)
//        {
//            shape[j] = A.shape[j];
//            shape2[j] = A.shape[j];
//        }
//        Tile<T> sumnorm(shape), avg_dev(shape2);
//        norm_sumnorm(A, sumnorm, i);
//        check_avg_dev(sumnorm, avg_dev, 1, eps0);
//        check_avg_dev(sumnorm, avg_dev, nelems, eps0);
//        check_avg_dev(sumnorm, avg_dev, nelems, eps1);
//        check_avg_dev(sumnorm, avg_dev, nelems, eps2);
//    }
//    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2}), -1, T{0}));
//    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2}), 0, T{0}));
//    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2}), 1, T{-1}));
//    TESTN(norm_avg_dev(Tile<T>({2}), Tile<T>({2}), 1, T{1}));
//    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({3}), 1, T{1}));
//    TESTN(norm_avg_dev(Tile<T>({3}), Tile<T>({2, 3}), 1, T{1}));
//    TESTN(norm_avg_dev(Tile<T>({}), Tile<T>({}), 1, T{1}));
//    TESTN(norm_avg_dev(Tile<T>({3, 4}), Tile<T>({2, 3}), 1, T{1}));
//}

int main(int argc, char **argv)
{
    //validate_avg_dev<fp32_t>();
    //validate_avg_dev<fp64_t>();
    return 0;
}

