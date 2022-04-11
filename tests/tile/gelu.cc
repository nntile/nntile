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
    std::vector<size_t> index(B.ndim, 0);
    copy(A, index, B, index);
    gelu(B);
    auto A_ptr = A.get_local_ptr(), B_ptr = B.get_local_ptr();
    for(size_t i = 0; i < B.nelems; ++i)
    {
        constexpr T sqrt2 = std::sqrt(T{2.0});
        constexpr T one = 1.0;
        constexpr T pt5 = 0.5;
        T val = A_ptr[i];
        T tmp = pt5*(std::erf(val/sqrt2)) + pt5;
        val *= tmp;
        T diff = abs(val - B_ptr[i]);
        T threshold = abs(val) * std::numeric_limits<T>::epsilon();
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
    T one = 1, zero = 0;
    std::vector<size_t> offset(A.ndim, 0);
    randn(A, offset, A.stride, seed, zero, one);
    check_gelu(A);
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_gelu<float>;
    validate_gelu<double>;
    return 0;
}

