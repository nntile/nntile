#include "nntile/tile/tile.hh"
#include "../testing.hh"

using namespace nntile;

template<typename T>
void check_tile_ptr(const TileTraits &traits)
{
    T *ptr = new T[traits.nelems];
    TESTN(Tile<T>(traits, ptr, traits.nelems-1));
    for(size_t i = 0; i < traits.nelems; ++i)
    {
        ptr[i] = static_cast<T>(i);
    }
    Tile<T> tile(traits, ptr, traits.nelems);
    TESTN(tile.at_linear(tile.nelems));
    for(size_t i = 0; i < tile.nelems; ++i)
    {
        T value = static_cast<T>(i);
        auto index = tile.linear_to_index(i);
        TESTA(value == tile.at_linear(i));
        TESTA(value == tile.at_index(index));
    }
    // Check tile with allocation done by StarPU
    Tile<T> tile2(traits);
    tile2.acquire(STARPU_W);
    TESTA(tile2.get_local_ptr() != nullptr);
    tile2.release();
    delete[] ptr;
}

template<typename T>
void validate_tile()
{
    // Check tiles with pointers
    check_tile_ptr<T>({{3, 2, 1}});
    check_tile_ptr<T>({{3, 2, 1}, {0, 0, 0}, {3, 2, 1}});
    check_tile_ptr<T>({{3, 2, 1}, {1, 1, 1}, {4, 3, 2}});
}

int main(int argc, char **argv)
{
    Starpu starpu;
    validate_tile<float>();
    validate_tile<double>();
    return 0;
}

