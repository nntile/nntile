#include "nntile/tile/copy.hh"
#include "nntile/tile/randn.hh"
#include "check_tiles_intersection.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void validate_copy()
{
    Tile<T> A({{4, 5, 6, 7}}), B({{2, 3, 4, 5}});
    unsigned long long seed = 100;
    randn(A, seed);
    copy_intersection_async(A, B);
    starpu_task_wait_for_all();
    TESTA(check_tiles_intersection(A, {0, 0, 0, 0}, B, {0, 0, 0, 0}) == 1);
    TESTA(check_tiles_intersection(A, {0, 0, 0, 0}, B, {1, 0, 0, 0}) == 0);
    copy_intersection(A, B);
    TESTA(check_tiles_intersection(A, {0, 0, 0, 0}, B, {0, 0, 0, 0}) == 1);
    TESTA(check_tiles_intersection(A, {0, 0, 0, 0}, B, {1, 0, 0, 0}) == 0);
    copy_intersection(A, {1, 2, 3, 4}, B, {2, 3, 4, 5});
    TESTA(check_tiles_intersection(A, {1, 2, 3, 4}, B, {2, 3, 4, 5}) == 1);
    TESTA(check_tiles_intersection(A, {1, 2, 3, 4}, B, {2, 3, 4, 4}) == 0);
    copy_intersection(A, {1, 2, 3, 4}, B, {0, 0, 2, 2});
    TESTA(check_tiles_intersection(A, {1, 2, 3, 4}, B, {0, 0, 2, 2}) == 1);
    TESTA(check_tiles_intersection(A, {1, 2, 3, 4}, B, {0, 0, 0, 0}) == 0);
    TESTA(check_tiles_intersection(A, B) == 0);
    copy_intersection(A, {1, 2, 3, 4}, B, {4, 5, 8, 0});
    TESTA(check_tiles_intersection(A, {1, 2, 3, 4}, B, {4, 5, 8, 0}) == 1);
    TESTA(check_tiles_intersection(A, {0, 2, 3, 4}, B, {3, 5, 8, 0}) == 1);
    TESTA(check_tiles_intersection(A, {0, 0, 3, 4}, B, {3, 3, 8, 0}) == 1);
    TESTA(check_tiles_intersection(A, {0, 0, 0, 4}, B, {3, 3, 5, 0}) == 1);
    copy_intersection(A, {1, 2, 3, 4}, B, {4, 5, 8, 11});
    TESTA(check_tiles_intersection(A, {1, 2, 3, 4}, B, {4, 5, 8, 11}) == 1);
    Tile<T> fail({{5}});
    TESTN(copy_intersection(A, fail));
    TESTN(copy_intersection(A, {0, 0, 0, 0}, B, {0, 0, 0}));
    TESTN(copy_intersection(A, {0, 0, 0}, B, {0, 0, 0, 0}));
    Tile<T> C({{4, 5, 6, 7}});
    randn(C, seed*seed+1);
    copy_intersection(B, {1, 0, 0, 0}, C, {1, 1, 1, 1});
    TESTA(check_tiles_intersection(B, {1, 0, 0, 0}, C, {1, 1, 1, 1}) == 1);
    TESTA(check_tiles_intersection(B, {1, 0, 0, 0}, A, {1, 1, 1, 1}) == 0);
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_copy<float>();
    validate_copy<double>();
    return 0;
}

