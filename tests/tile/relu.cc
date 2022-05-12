#include "nntile/tile/relu.hh"
#include "nntile/tile/randn.hh"
#include "nntile/tile/copy.hh"
#include <limits>
#include <cmath>

using namespace nntile;

template<typename T>
void check_relu(const Tile<T> &A)
{
    Tile<T> B(A.shape);
    std::vector<Index> index(B.ndim);
    copy_intersection(A, index, B, index);
    relu(B);
    A.acquire(STARPU_R);
    B.acquire(STARPU_R);
    auto A_ptr = A.get_local_ptr(), B_ptr = B.get_local_ptr();
    for(Index i = 0; i < B.nelems; ++i)
    {
        T val = std::max(T{0}, A_ptr[i]);
        T diff = std::abs(val - B_ptr[i]);
        T threshold = std::abs(val) * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            A.release();
            B.release();
            throw std::runtime_error("diff > threshold");
        }
    }
    A.release();
    B.release();
}

template<typename T>
void validate_relu()
{
    Tile<T> scalar({}), A({4, 5, 6, 3});
    unsigned long long seed = 100000000000001ULL;
    randn(scalar, seed);
    check_relu(scalar);
    randn(A, seed);
    check_relu(A);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_relu<float>();
    validate_relu<double>();
    return 0;
}

