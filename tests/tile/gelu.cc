#include "nntile/tile/gelu.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"
#include <limits>
#include <cmath>

using namespace nntile;

template<typename T>
void check_gelu(const Tile<T> &A)
{
    Tile<T> B(A.shape);
    std::vector<Index> index(B.ndim);
    copy_intersection(A, index, B, index);
    gelu(B);
    auto A_local = A.acquire(STARPU_R), B_local = B.acquire(STARPU_R);
    for(Index i = 0; i < B.nelems; ++i)
    {
        constexpr T one = 1.0;
        constexpr T pt5 = 0.5;
        const T sqrt2 = std::sqrt(T{2.0});
        T val = A_local[i];
        T tmp = pt5*(std::erf(val/sqrt2)) + pt5;
        val *= tmp;
        T diff = std::abs(val - B_local[i]);
        T threshold = std::abs(val) * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            throw std::runtime_error("diff > threshold");
        }
    }
}

template<typename T>
void validate_gelu()
{
    Tile<T> A({4, 5, 6, 3});
    unsigned long long seed = 100;
    randn(A, seed);
    check_gelu(A);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_gelu<float>();
    validate_gelu<double>();
    return 0;
}

