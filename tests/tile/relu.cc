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
    auto A_local = A.acquire(STARPU_R), B_local = B.acquire(STARPU_R);
    for(Index i = 0; i < B.nelems; ++i)
    {
        T val = std::max(T{0}, A_local[i]);
        T diff = std::abs(val - B_local[i]);
        T threshold = std::abs(val) * std::numeric_limits<T>::epsilon();
        if(diff > threshold)
        {
            throw std::runtime_error("diff > threshold");
        }
    }
}

template<typename T>
void validate_relu()
{
    Tile<T> scalar({}), A({40, 50, 60, 30});
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

