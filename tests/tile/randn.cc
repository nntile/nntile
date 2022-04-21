#include "nntile/tile/randn.hh"
#include "check_tiles_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tile<T> scalar({}), scalar2({});
    T one = 1, zero = 0;
    constexpr unsigned long long seed = 100000000000001ULL;
    randn(scalar, {}, {}, {}, seed);
    randn(scalar2, {}, {}, {}, seed);
    TESTA(check_tiles_intersection(scalar, scalar2));
    randn(scalar2, seed*seed);
    TESTA(!check_tiles_intersection(scalar, scalar2));
    Tile<T> big({5, 5, 5, 5}), small({2, 2, 2, 2});
    randn_async(big, seed);
    starpu_task_wait_for_all();
    randn(small, {1, 2, 3, 2}, big.shape, big.stride, seed);
    TESTA(check_tiles_intersection(big, {0, 0, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(check_tiles_intersection(small, {1, 2, 3, 2}, big, {0, 0, 0, 0}));
    TESTA(!check_tiles_intersection(big, {1, 0, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tiles_intersection(big, {0, 1, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tiles_intersection(big, {0, 0, 1, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tiles_intersection(big, {0, 0, 0, 1}, small, {1, 2, 3, 2}));
    TESTA(!check_tiles_intersection(small, {1, 2, 3, 2}, big, {1, 0, 0, 0}));
    TESTA(!check_tiles_intersection(small, {1, 2, 3, 2}, big, {0, 1, 0, 0}));
    TESTA(!check_tiles_intersection(small, {1, 2, 3, 2}, big, {0, 0, 1, 0}));
    TESTA(!check_tiles_intersection(small, {1, 2, 3, 2}, big, {0, 0, 0, 1}));
    TESTN(randn(small, {4, 0, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 4, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 4, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 0, 4}, big.shape, big.stride, seed));
    TESTN(randn(small, {-1, 0, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, -1, 0, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, -1, 0}, big.shape, big.stride, seed));
    TESTN(randn(small, {0, 0, 0, -1}, big.shape, big.stride, seed));
    auto stride = big.stride;
    ++stride[0];
    TESTN(randn(small, {0, 0, 0, 0}, big.shape, stride, seed));
    for(int i = 1; i < stride.size(); ++i)
    {
        --stride[i-1];
        ++stride[i];
        TESTN(randn(small, {0, 0, 0, 0}, big.shape, stride, seed));
    }
    small.acquire(STARPU_RW);
    const_cast<T *>(small.get_local_ptr())[small.nelems-1] = 0;
    small.release();
    TESTA(!check_tiles_intersection(big, {0, 0, 0, 0}, small, {1, 2, 3, 2}));
    TESTA(!check_tiles_intersection(small, {1, 2, 3, 2}, big, {0, 0, 0, 0}));
    Tile<T> small2({3, 3, 3, 3});
    randn(small2, {1, 1, 1, 1}, big.shape, big.stride, seed);
    TESTA(check_tiles_intersection(small, {1, 2, 3, 2}, small2, {1, 1, 1, 1}));
    TESTA(check_tiles_intersection(small2, {1, 1, 1, 1}, small, {1, 2, 3, 2}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

