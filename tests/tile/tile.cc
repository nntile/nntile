#include <nntile/tile/tile.hh>

using namespace nntile;

template<typename T>
void validate_tile()
{
    // Traits for different tiles to check operations
    TileTraits A1_traits({3, 2, 1, 10}),
               A1T_traits({10, 3, 2, 1}),
               B1_traits({10, 5, 6}),
               B1T_traits({5, 6, 10}),
               C1_traits({3, 2, 1, 5, 6}),
               C1T_traits({5, 6, 3, 2, 1}),
               A2_traits({3, 4, 5}),
               A2T_traits({4, 5, 3}),
               B2_traits({4, 5, 5, 6}),
               B2T_traits({5, 6, 4, 5}),
               C2_traits({3, 5, 6}),
               C2T_traits({5, 6, 3});
    // Check tile without allocating memory
    Tile<T> tmp(A1_traits);
    // Allocate memory for tiles
    auto *A1_ptr = new T[A1_traits.nelems];
    auto *B1_ptr = new T[B1_traits.nelems];
    auto *C1_ptr = new T[C1_traits.nelems];
    auto *A2_ptr = new T[A2_traits.nelems];
    auto *B2_ptr = new T[B2_traits.nelems];
    auto *C2_ptr = new T[C2_traits.nelems];
    // Construct tiles
    Tile<T> A1(A1_traits, A1_ptr, A1_traits.nelems),
        A1T(A1T_traits, A1_ptr, A1_traits.nelems),
        B1(B1_traits, B1_ptr, B1_traits.nelems),
        B1T(B1T_traits, B1_ptr, B1_traits.nelems),
        C1(C1_traits, C1_ptr, C1_traits.nelems),
        C1T(C1T_traits, C1_ptr, C1_traits.nelems),
        A2(A2_traits, A2_ptr, A2_traits.nelems),
        A2T(A2T_traits, A2_ptr, A2_traits.nelems),
        B2(B2_traits, B2_ptr, B2_traits.nelems),
        B2T(B2T_traits, B2_ptr, B2_traits.nelems),
        C2(C2_traits, C2_ptr, C2_traits.nelems),
        C2T(C2T_traits, C2_ptr, C2_traits.nelems);
}

int main(int argc, char **argv)
{
    StarPU starpu;
    validate_tile<float>();
    validate_tile<double>();
    return 0;
}

