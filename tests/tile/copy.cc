#include "nntile/tile/copy.hh"
#include "nntile/tile/randn.hh"
#include "check_tiles_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_copy()
{
    Tile<T> A({4, 5, 6, 7}), B({2, 3, 4, 5});
    unsigned long long seed = 100;
    randn(A, seed);
    copy_async(A, {0, 0, 0, 0}, B, {0, 0, 0, 0});
    starpu_task_wait_for_all();
    check_tiles_intersection(A, {0, 0, 0, 0}, B, {0, 0, 0, 0});
    copy(A, {1, 2, 3, 4}, B, {2, 3, 4, 5});
    check_tiles_intersection(A, {1, 2, 3, 4}, B, {2, 3, 4, 5});
    copy(A, {1, 2, 3, 4}, B, {0, 0, 2, 2});
    check_tiles_intersection(A, {1, 2, 3, 4}, B, {0, 0, 2, 2});
    copy(A, {1, 2, 3, 4}, B, {4, 5, 8, 0});
    check_tiles_intersection(A, {1, 2, 3, 4}, B, {4, 5, 8, 0});
    copy(A, {1, 2, 3, 4}, B, {4, 5, 8, 11});
    check_tiles_intersection(A, {1, 2, 3, 4}, B, {4, 5, 8, 11});
    Tile<T> fail({5});
    TESTN(copy(A, {0, 0, 0, 0}, fail, {0, 0, 0, 0}));
    TESTN(copy(A, {0, 0, 0, 0}, B, {0, 0, 0}));
    TESTN(copy(A, {0, 0, 0}, B, {0, 0, 0, 0}));
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_copy<float>();
    validate_copy<double>();
    return 0;
}

