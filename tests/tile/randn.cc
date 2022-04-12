#include "nntile/tile/randn.hh"
#include "check_tiles_intersection.hh"
#include <iostream>

using namespace nntile;

template<typename T>
void validate_randn()
{
    Tile<T> big({5, 5, 5, 5}), big2({5, 5, 5, 5}), small({2, 2, 2, 2}),
        small2({3, 3, 3, 3});
    T one = 1, zero = 0;
    constexpr unsigned long long seed = 100;
    randn(big, seed, zero, one);
    randn(big, big2, {0, 0, 0, 0}, seed, zero, one);
    check_tiles_intersection(big2, {0, 0, 0, 0}, big, {0, 0, 0, 0});
    randn(big, small, {0, 1, 2, 3}, seed, zero, one);
    check_tiles_intersection(small, {0, 1, 2, 3}, big, {0, 0, 0, 0});
    randn(big, small2, {0, 2, 0, 1}, seed, zero, one);
    check_tiles_intersection(small, {0, 1, 2, 3}, small2, {0, 2, 0, 1});
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_randn<float>();
    validate_randn<double>();
    return 0;
}

