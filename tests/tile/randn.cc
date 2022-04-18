#include "nntile/tile/randn.hh"
#include "check_tiles_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tile<T> scalar({{}}), scalar2({{}});
    T one = 1, zero = 0;
    constexpr unsigned long long seed = 100000000000001ULL;
    randn(scalar, seed);
    randn(scalar2, seed);
    TESTA(check_tiles_intersection(scalar, scalar2) == 1);
    randn(scalar2, seed*seed);
    TESTA(check_tiles_intersection(scalar, scalar2) == 0);
    Tile<T> big({{5, 5, 5, 5}}),
        small({{2, 2, 2, 2}, {0, 1, 2, 3}, {5, 5, 5, 5}});
    randn(big, seed);
    randn(small, seed);
    TESTA(check_tiles_intersection(big, small) == 1);
    TESTA(check_tiles_intersection(small, big) == 1);
    small.acquire(STARPU_RW);
    const_cast<T *>(small.get_local_ptr())[small.nelems-1] = 0;
    small.release();
    TESTA(check_tiles_intersection(big, small) == 0);
    TESTA(check_tiles_intersection(small, big) == 0);
    Tile<T> small2({{3, 3, 3, 3}, {1, 2, 2, 1}, {5, 5, 5, 5}});
    randn(small2, seed);
    TESTA(check_tiles_intersection(small, small2) == 1);
    TESTA(check_tiles_intersection(small2, small) == 1);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

