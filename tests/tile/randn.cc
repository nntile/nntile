#include "nntile/tile/randn.hh"
#include "check_tiles_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tile<T> big({5, 5, 5, 5}), big2({5, 5, 5, 5}), small({2, 2, 2, 2}),
        small2({3, 3, 3, 3}), scalar({});
    T one = 1, zero = 0;
    constexpr unsigned long long seed = 100;
    randn(scalar, seed);
    randn_async(big, seed, zero, one);
    starpu_task_wait_for_all();
    randn(big, big2, {0, 0, 0, 0}, seed, zero, one);
    check_tiles_intersection(big2, {0, 0, 0, 0}, big, {0, 0, 0, 0});
    randn(big, small, {0, 1, 2, 3}, seed, zero, one);
    check_tiles_intersection(small, {0, 1, 2, 3}, big, {0, 0, 0, 0});
    randn(big, small2, {0, 2, 0, 1}, seed, zero, one);
    check_tiles_intersection(small, {0, 1, 2, 3}, small2, {0, 2, 0, 1});
    randn(big, big2, {5, 0, 0, 0}, seed);
    TESTN(randn(big, big, {0, 0, 0}, seed));
    TESTN(randn(big, big2, {0, 0, 0, 0, 0}, seed));
    Tile<T> fail({5});
    TESTN(randn(big, fail, {0, 0, 0, 0}, seed));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

